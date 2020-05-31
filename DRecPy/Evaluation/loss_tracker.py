import matplotlib.pyplot as plt


class LossTracker:
    def __init__(self):
        self.losses = []
        self.curr_avg_loss = 0

        self.epoch_losses = []
        self.curr_avg_epoch_loss = 0

        self.epoch_callback_results = {}
        self.called_epochs = []

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

    def add_epoch_callback_result(self, name, result, epoch):
        if name not in self.epoch_callback_results:
            self.epoch_callback_results[name] = []

        self.epoch_callback_results[name].append(result)
        if len(self.called_epochs) == 0 or self.called_epochs[-1] < epoch:
            self.called_epochs.append(epoch)

    def display_graph(self, model_name=None):
        if len(self.epoch_callback_results) > 0:
            fig, axes = plt.subplots(nrows=2, ncols=1)
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1)
            axes = [axes]

        if model_name is None:
            fig.suptitle('Loss per Epoch')
        else:
            fig.suptitle(f'Loss per Epoch of {model_name}')

        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].plot(self.epoch_losses)

        if len(axes) > 1:
            axes[1].set_ylabel("Value", fontsize=12)
            for k in self.epoch_callback_results:
                axes[1].plot(self.called_epochs, self.epoch_callback_results[k], label=k)

        axes[-1].set_xlabel("Epoch", fontsize=12)

        plt.legend()
        plt.show(block=block)
