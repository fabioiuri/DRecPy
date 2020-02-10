import matplotlib.pyplot as plt


class LossTracker:
    def __init__(self):
        self.losses = []
        self.curr_avg_loss = 0

        self.epoch_losses = []
        self.curr_avg_epoch_loss = 0

    def add_batch_loss(self, loss):
        self.losses.append(loss)
        self.curr_avg_loss = self.curr_avg_loss + (loss - self.curr_avg_loss) / len(self.losses)

    def get_batch_avg_loss(self):
        return self.curr_avg_loss

    def reset_batch_losses(self):
        self.losses = []
        self.curr_avg_loss = 0

    def update_epoch_loss(self):
        self.epoch_losses.append(self.curr_avg_loss)
        self.curr_avg_epoch_loss = self.curr_avg_epoch_loss + \
                                   (self.curr_avg_loss - self.curr_avg_epoch_loss) / len(self.epoch_losses)

    def reset_epoch_losses(self):
        self.epoch_losses = []
        self.curr_avg_epoch_loss = 0

    def display_graph(self, model_name=None):
        fig, axes = plt.subplots(1)
        if model_name is None: fig.suptitle('Loss per Epoch')
        else: fig.suptitle(f'Loss per Epoch of {model_name}')

        axes.set_ylabel("Loss", fontsize=12)
        axes.set_xlabel("Epoch", fontsize=12)
        axes.plot(self.epoch_losses)
        plt.show()
