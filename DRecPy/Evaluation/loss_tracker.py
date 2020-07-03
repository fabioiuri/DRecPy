import matplotlib.pyplot as plt


class LossTracker:
    def __init__(self):
        """A LossTracker instance that tracks epoch losses, computed average epoch losses, and
        also tracks epoch callback losses. This utility object is used by obj:`DRecPy.Recommender.RecommenderABC` to
        track all the losses during the training process."""
        self.epoch_losses = []
        self.curr_avg_epoch_loss = 0

        self.epoch_callback_results = {}
        self.called_epochs = []

    def add_epoch_loss(self, loss):
        """Adds a new epoch loss.

        Args:
            loss: The loss value obtained during the epoch.
        """
        self.epoch_losses.append(loss)
        self.curr_avg_epoch_loss = self.curr_avg_epoch_loss + (loss - self.curr_avg_epoch_loss) / len(self.epoch_losses)

    def get_epoch_avg_loss(self):
        """Gets the current average epoch loss.

        Returns:
            The current average epoch loss, computed from the provided epoch losses.
        """
        return self.curr_avg_epoch_loss

    def reset_epoch_losses(self):
        """Resets the stored epoch losses and sets the epoch average loss to 0."""
        self.epoch_losses = []
        self.curr_avg_epoch_loss = 0

    def add_epoch_callback_result(self, name, result, epoch):
        """Adds a new epoch callback result.

        Args:
            name: A string representing the name of the evaluated metric.
            result: A number representing the evaluated value for the current metric.
            epoch: A number representing the epoch for which the evaluated metric showed the passed result.
        """
        if name not in self.epoch_callback_results:
            self.epoch_callback_results[name] = []

        self.epoch_callback_results[name].append(result)
        if len(self.called_epochs) == 0 or self.called_epochs[-1] < epoch:
            self.called_epochs.append(epoch)

    def display_graph(self, model_name=None, block=False):
        """Displays a graph containing the average batch loss per epoch, as well as the epoch callback results for
        each metric, if they exist.

        Args:
            model_name: A string representing the name of the model. If this is provided, the model name will be
                displayed on the title of the figure. Default: None.
            block: A boolean indicating whether the displayed graph should block code execution or not. Default: False.
        """
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
            for m in self.epoch_callback_results:
                try:
                    axes[1].plot(self.called_epochs, self.epoch_callback_results[m], label=m)
                except ValueError:
                    raise Exception(f'Epoch callback results for metric {m} are not defined for all called epochs: '
                                    f'number of called epochs: {len(self.called_epochs)}, number of epoch callback '
                                    f'results for metric {m}: {len(self.epoch_callback_results[m])}')

        axes[-1].set_xlabel("Epoch", fontsize=12)

        plt.legend()
        plt.show(block=block)
