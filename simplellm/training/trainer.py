class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        dataloaders,
        config
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.config = config