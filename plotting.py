import matplotlib.pyplot as plt
import seaborn as sns


def plot_history(history, figsize=(14, 5)):
    """Plot train/test loss and accuracy curves.
    """
    acc = history.train_acc
    val_acc = history.val_acc
    loss = history.train_loss
    val_loss = history.val_loss
    epochs = range(1, len(acc) + 1)

    # Accuracy plot
    fig, ax = plt.subplots(1, 2, figsize=figsize, tight_layout=True)
    ax[0].plot(epochs, acc, label="Training acc")
    ax[0].plot(epochs, val_acc, label="Validation acc")
    ax[0].set_title("Training and Validation accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    # Loss plot
    ax[1].plot(epochs, loss, label="Training loss")
    ax[1].plot(epochs, val_loss, label="Validation loss")
    ax[1].set_title("Training and Validation loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    return fig, ax


def plot_cmatrix(cmatrix, label_names, **heatmap_kwargs):
    """Plot confusion matrix
    """
    fig, ax = plt.subplots()
    sns.heatmap(cmatrix, ax=ax, **heatmap_kwargs)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticklabels(label_names, fontsize=12)
    ax.set_yticklabels(label_names, fontsize=12)
    return fig, ax
