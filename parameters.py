"""Model, optimization, and output settings for the MNIST classroom example."""

SEED = 42
INPUT_DIM = 784
HIDDEN_DIM = 64
OUTPUT_DIM = 10
BATCH_SIZE = 128
EPOCHS = 100  # Maximum; training may stop earlier.

SGD_LR = 0.01
ADAM_DEFAULT_LR = 0.001
ADAM_DEFAULT_BETAS = (0.9, 0.999)
ADAM_GRID = {
    "beta1": [0.85, 0.9, 0.95],
    "beta2": [0.99, 0.995, 0.999],
}

LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 10

DATA_PATH = "data/mnist.pkl.gz"
RESULTS_DIR = "results"
