import os
import random
import time
import torch
from torch.utils.data import DataLoader

from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim

import torchvision
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train(model_path):
	writer = SummaryWriter(log_dir='/mnt/artifacts/results/runs_norm_vox1_vox2')
	device = torch.device(hp.device)

	if hp.data.data_preprocessed:
	    train_dataset = SpeakerDatasetTIMITPreprocessed(task='train')
	    test_dataset = SpeakerDatasetTIMITPreprocessed(task='test')

	else:
	    train_dataset = SpeakerDatasetTIMIT()
	    test_dataset = SpeakerDatasetTIMIT()

	train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=True) 
	test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)    

	embedder_net = SpeechEmbedder().to(device)

	if hp.train.restore:
	    embedder_net.load_state_dict(torch.load(model_path))
	    epoch = int(model_path.split('epoch_')[1].split('_')[0])
	else:
		epoch = 0
	ge2e_loss = GE2ELoss(device)
	#Both net and loss have trainable parameters
	optimizer = torch.optim.SGD([
	                {'params': embedder_net.parameters()},
	                {'params': ge2e_loss.parameters()}
	            ], lr=hp.train.lr)

	os.makedirs(hp.train.checkpoint_dir, exist_ok=True)

	embedder_net.train()

	steps = 0
	running_loss = 0
	print_every = 20
	avg_train_loss_per_epoch = 0

	for epoch in range(epoch,hp.train.epochs):
		avg_train_loss_per_epoch /= len(train_loader)
		writer.add_scalar('avg_train_loss_per_epoch',avg_train_loss_per_epoch,epoch)
		avg_train_loss_per_epoch = 0


		for batch_id, mel_db_batch in enumerate(train_loader):
# 			import pdb; pdb.set_trace()
			steps += 1
			# Move input and label tensors to the default device
			mel_db_batch = mel_db_batch.to(device)

			mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
			perm = random.sample(range(0, hp.train.N*hp.train.M), hp.train.N*hp.train.M)
			unperm = list(perm)
			for i,j in enumerate(perm):
				unperm[j] = i
			mel_db_batch = mel_db_batch[perm]


			optimizer.zero_grad()

			embeddings = embedder_net(mel_db_batch)
			embeddings = embeddings[unperm]
			embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))

			#get loss, call backward, step optimizer
			loss = ge2e_loss(embeddings) #wants (Speaker, Utterances, embedding)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
			torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)

			optimizer.step()

			running_loss += loss.item()
			avg_train_loss_per_epoch += loss.item()
			writer.add_scalar('train loss at each step',loss.item(),steps)

			if steps % print_every == 0:
			    test_loss = 0

			    embedder_net.eval()
			    with torch.no_grad():
			    	for test_batch_id, test_mel_db_batch in enumerate(test_loader):
			    		test_mel_db_batch = test_mel_db_batch.to(device)

			    		test_mel_db_batch = torch.reshape(test_mel_db_batch, (hp.train.N*hp.train.M, test_mel_db_batch.size(2), test_mel_db_batch.size(3)))
			    		perm = random.sample(range(0, hp.train.N*hp.train.M), hp.train.N*hp.train.M)
			    		unperm = list(perm)

			    		for i,j in enumerate(perm):
			    			unperm[j] = i
			    		test_mel_db_batch = test_mel_db_batch[perm]
			        
			    		test_embeddings = embedder_net(test_mel_db_batch)
			    		test_embeddings = test_embeddings[unperm]
			    		test_embeddings = torch.reshape(test_embeddings, (hp.train.N, hp.train.M, test_embeddings.size(1)))
		            
			            #get loss, call backward, step optimizer
			    		batch_loss = ge2e_loss(test_embeddings) #wants (Speaker, Utterances, embedding)

			    		test_loss += batch_loss.item()
			    		writer.add_scalar('test loss at each test iteration',batch_loss.item(),steps)

			            
			    print(f"Epoch {epoch+1}/{hp.train.epochs}.. "
			          f"Train loss: {running_loss/print_every:.3f}.. "
			          f"Test loss: {test_loss/len(test_loader):.3f}.. ")

			    writer.add_scalar('running_loss every 5 steps',running_loss/print_every, steps)
			    writer.add_scalar('test_loss',test_loss/len(test_loader),steps)

			    running_loss = 0
			    embedder_net.train()

		if hp.train.checkpoint_dir is not None and (epoch + 1) % hp.train.checkpoint_interval == 0:
			embedder_net.eval().cpu()
			ckpt_model_filename = "ckpt_epoch_" + str(epoch+1) + "_batch_id_" + str(batch_id+1) + \
								   datetime.now().strftime("%d-%b-%Y_%H:%M:%S.%f") +".pth"
			ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
			torch.save(embedder_net.state_dict(), ckpt_model_path)
			embedder_net.to(device).train()

	#save model
	embedder_net.eval().cpu()
	save_model_filename = "final_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".model"
	save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
	torch.save(embedder_net.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)

	writer.close()

train(hp.model.model_path)
