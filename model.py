"""One-hidden-layer network and reusable MNIST data, evaluation, and plotting operations."""

import gzip
import pickle

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class OneHiddenLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        a = self.hidden(x)
        h = self.activation(a)
        logits = self.output(h)
        return logits


def load_mnist(path):
    """Load the conventional (train, validation, test) mnist.pkl.gz splits."""
    with gzip.open(path, "rb") as file:
        splits = pickle.load(file, encoding="latin1")
    datasets = []
    for images, labels in splits:
        x = torch.as_tensor(images, dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.long)
        datasets.append(TensorDataset(x, y))
    return tuple(datasets)


def make_loaders(train_data, valid_data, test_data, batch_size, seed, pin_memory):
    """Use identical shuffled batch ordering for each independent training run."""
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              generator=generator, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=device.type == "cuda")
            y_batch = y_batch.to(device, non_blocking=device.type == "cuda")
            total_loss += loss_fn(model(x_batch), y_batch).item() * len(y_batch)
    return total_loss / len(loader.dataset)


def test_predictions(model, loader, device, n_classes):
    """Return held-out accuracy and an observed-by-predicted confusion matrix."""
    model.eval()
    confusion = torch.zeros(n_classes * n_classes, dtype=torch.int64)
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device, non_blocking=device.type == "cuda"))
            predicted = logits.argmax(dim=1).cpu()
            confusion += torch.bincount(y_batch * n_classes + predicted,
                                        minlength=n_classes * n_classes)
    matrix = confusion.reshape(n_classes, n_classes)
    return matrix.diag().sum().item() / matrix.sum().item(), matrix


def plot_losses(histories, filename, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, history in histories.items():
        ax.plot(range(1, len(history["valid_loss"]) + 1), history["valid_loss"], label=name)
    ax.set(xlabel="Epoch", ylabel="Validation cross-entropy", title=title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.show()
    plt.close(fig)


def plot_confusion(matrix, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.numpy(), cmap="Blues")
    ax.set(xlabel="Predicted digit", ylabel="True digit", title="Tuned Adam: test confusion matrix")
    ax.set_xticks(range(matrix.shape[0]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.show()
    plt.close(fig)


def plot_digit_probabilities_grid(model, dataset, indices, device, n_classes, filename, ncols=2):
    """Plot test digits beside their predicted class probabilities."""
    images = torch.stack([dataset[i][0] for i in indices])
    actual = [int(dataset[i][1]) for i in indices]
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    nrows = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, 2 * ncols, figsize=(14, 3.2 * nrows),
                             gridspec_kw={"width_ratios": [1, 2] * ncols},
                             squeeze=False)
    for i, (image, truth, probs) in enumerate(zip(images, actual, probabilities)):
        row, col = divmod(i, ncols)
        image_ax, bar_ax = axes[row, 2 * col:2 * col + 2]
        prediction = int(probs.argmax())
        image_ax.imshow(image.reshape(28, 28).numpy(), cmap="gray")
        image_ax.set_title(f"True: {truth} | Predicted: {prediction}")
        image_ax.axis("off")
        bar_ax.barh(range(n_classes), probs)
        bar_ax.set(xlim=(0, 1), yticks=range(n_classes),
                   xlabel="Predicted probability", title=f"Test observation {indices[i]}")
        bar_ax.invert_yaxis()
    for i in range(len(indices), nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, 2 * col].axis("off")
        axes[row, 2 * col + 1].axis("off")
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.show()
    plt.close(fig)
