import ray
from ray.util.sgd import TorchTrainer
from train_speech_embedder_norm import train

ray.init()

trainer = TorchTrainer(
    training_operator_cls=train,
    config={"lr": 0.001},
    num_workers=100,
    use_gpu=True)

for i in range(10):
    metrics = trainer.train()
    val_metrics = trainer.validate()
