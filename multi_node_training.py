from ray.util.sgd.torch import TorchTrainer, TrainingOperator

def train(*, model=None, criterion=None, optimizer=None, dataloader=None):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return {
        "accuracy": correct / total,
        "train_loss": train_loss / (batch_idx + 1)
    }

def model_creator(config):
    return Discriminator(), Generator()

def optimizer_creator(models, config):
    net_d, net_g = models
    discriminator_opt = optim.Adam(
        net_d.parameters(), lr=config.get("lr", 0.01), betas=(0.5, 0.999))
    generator_opt = optim.Adam(
        net_g.parameters(), lr=config.get("lr", 0.01), betas=(0.5, 0.999))
    return discriminator_opt, generator_opt

class CustomOperator(TrainingOperator):
    def setup(self, config):
        net_d = Discriminator()
        net_g = Generator()

        d_opt = optim.Adam(
            net_d.parameters(), lr=config.get("lr", 0.01), betas=(0.5, 0.999))
        g_opt = optim.Adam(
            net_g.parameters(), lr=config.get("lr", 0.01), betas=(0.5, 0.999))

        # Setup data loaders, loss, schedulers here.
        ...

        # Register all the components.
        self.models, self.optimizers, ... = self.register(models=(net_d, net_g), optimizers=(d_opt, g_opt), ...)
        self.register_data(...)

    def train_epoch(self, iterator, info):
        result = {}
        for i, (model, optimizer) in enumerate(
                zip(self.models, self.optimizers)):
            result["model_{}".format(i)] = train(
                model=model,
                criterion=self.criterion,
                optimizer=optimizer,
                dataloader=iterator)
        return result

trainer = TorchTrainer(training_operator_cls=CustomOperator)

stats = trainer.train()
