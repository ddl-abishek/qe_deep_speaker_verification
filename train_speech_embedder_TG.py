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

import ray
from ray.util.sgd.torch import TorchTrainer, TrainingOperator


def model_creator():
    device = torch.device(hp.device)
    embedder_net = SpeechEmbedder().to(device)
    
    return embedder_net

def optimizer_creator(device,embedder_net):
    ge2e_loss = GE2ELoss(device)
    
    optimizer = torch.optim.SGD([
                    {'params': embedder_net.parameters()},
                    {'params': ge2e_loss.parameters()}
                ], lr=hp.train.lr)
    
    return optimizer


def data_creator():
    train_dataset = SpeakerDatasetTIMITPreprocessed(task='train')
    test_dataset = SpeakerDatasetTIMITPreprocessed(task='test')
    
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=True) 
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)    

    return train_loader, test_loader

def scheduler_creator(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.1, gamma=0.9)

def loss_creator(device,train_loader,optimizer,embedder_net,ge2e_loss):
    for batch_id, mel_db_batch in enumerate(train_loader):
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
        return loss

CustomTrainingOperator = TrainingOperator.from_creators(model_creator=model_creator,
                                                        optimizer_creator=optimizer_creator,
                                                        data_creator=data_creator, 
                                                        scheduler_creator=scheduler_creator,
                                                        loss_creator=loss_creator)
if ray.is_initialized() == False:
        print("Connecting to Ray cluster...")
        service_host = os.environ["RAY_HEAD_SERVICE_HOST"]
        service_port = os.environ["RAY_HEAD_SERVICE_PORT"]
        ray.util.connect(f"{service_host}:{service_port}")

trainer = TorchTrainer(training_operator_cls=CustomTrainingOperator,
                       num_workers=3,
                       use_gpu=False,
                       config={"lr": 1e-2,"batch_size": hp.train.N*hp.train.M},
                       scheduler_step_freq="epoch")
trainer.train()