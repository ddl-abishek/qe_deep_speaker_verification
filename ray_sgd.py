import ray
from ray.util.sgd import TorchTrainer

ray.init()

trainer = TorchTrainer(
    training_operator_cls=MyTrainingOperator,
    config={"lr": 0.001},
    num_workers=100,
    use_gpu=True)

for i in range(10):
    metrics = trainer.train()
    val_metrics = trainer.validate()
