class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        dataloaders,
        callbacks,
        config
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.callbacks = callbacks
        self.config = config

    def fit(self, NUM_EPOCHS):
        ''' adopting general function outline from MosaicML's Composer library '''
        # <INIT>
        # <AFTER_LOAD>
        # <FIT_START>
        for epoch in range(NUM_EPOCHS):
            # <EPOCH_START>
            while True:
                # <BEFORE_DATALOADER>
                batch = next(self.dataloader)
                if batch is None:
                    break
                inputs, targets = batch
                # <AFTER_DATALOADER>

                # <BATCH_START>

                # <BEFORE_FORWARD>
                outputs = self.model(inputs)
                # <AFTER_FORWARD>

                # <BEFORE_LOSS>
                loss = self.loss(outputs, targets)
                # <AFTER_LOSS>

                # <BEFORE_BACKWARD>
                loss.backward()
                # <AFTER_BACKWARD>

                self.optimizer.step()
                self.optimizer.zero_grad()

                # <BATCH_END>
            # <EPOCH_END>