# A Gentle Introduction to Neural Networks: MNIST Example

This repository accompanies the lecture *A Gentle Introduction to Neural Networks for Predictive Modeling*.

In this example, we will build and train a simple neural network using Python and PyTorch to predict handwritten digits from a small subset of the MNIST dataset.

The goal is to demonstrate how the mathematical concepts introduced in the lecture translate into practical code. In particular, we will examine:

- Constructing a neural network using linear layers and activation functions.
- Training a neural network through backpropagation.
- Using mini-batches to estimate gradients efficiently.
- Updating model parameters using gradient-based optimization.
- Organizing Python files and Jupyter notebooks within a reproducible project.
- Using Git and GitHub to save and share code.

We will use Visual Studio Code (VS Code) as our development environment. Although other integrated development environments (IDEs) are available, all instructions will be demonstrated using VS Code on Windows.

No prior experience with PyTorch is required, although some familiarity with Python will be helpful.

---

## Section 1: Downloading the Necessary Tools

Before beginning, we need to install several programs that will allow us to write, execute, and save our neural network code.

### 1.1 Installing Python

Python is the programming language we will use to construct and train our neural network.

1. Visit the official Python website: https://www.python.org/downloads/
2. Download a compatible version of Python for your operating system.
3. Run the installer and follow the installation instructions.
4. On Windows, ensure Python is added to your system PATH if the installer offers this option.

Once installed, open a terminal and enter:

```bash
python --version
```

This should display your installed Python version.

### 1.2 Jupyter Notebooks

Jupyter notebooks allow us to execute Python code in individual cells, inspect intermediate calculations, and display results without running an entire Python script.

We will use Jupyter notebooks directly within VS Code, so a separate Jupyter desktop application is not required.

We will install the necessary Python packages in Section 4.

For additional information, visit https://jupyter.org/.

### 1.3 Installing Git

Git is a version control system that allows us to track changes to our code and save our work in repositories.

Toward the end of the example, we will briefly demonstrate how to create a Git repository and upload our project to GitHub directly through the VS Code terminal.

1. Visit https://git-scm.com/downloads
2. Download Git for your operating system.
3. Follow the installation instructions.

To verify the installation, enter:

```bash
git --version
```

### 1.4 Downloading the MNIST Dataset

MNIST is a dataset containing grayscale images of handwritten digits from 0 through 9.

For this example, we will use a small subset of MNIST to keep training times short and the code easy to understand.

Download the dataset from:

**[MNIST dataset download link — to be provided]**

Save the downloaded file somewhere accessible on your computer. We will move it into the appropriate project folder in Section 3.

### 1.5 Creating Your Project Folder

Create a new folder anywhere on your computer and name it:

```text
MNIST_NeuralNet_Example_last-name-here
```

Replace `last-name-here` with your own last name.

This folder will contain our Python scripts, Jupyter notebook, dataset, and saved results.

---

## Section 2: Selecting an IDE

An integrated development environment (IDE) provides tools for writing, organizing, and executing code.

Although there are many available options, we will use **Visual Studio Code (VS Code)** because it supports Python, Jupyter notebooks, integrated terminals, and Git.

### 2.1 Installing VS Code

1. Visit https://code.visualstudio.com/
2. Download VS Code for your operating system.
3. Follow the installation instructions.
4. Open VS Code once installation is complete.

### 2.2 Installing VS Code Extensions

VS Code supports extensions that provide additional functionality.

On the left sidebar, select the **Extensions** icon, or press `Ctrl + Shift + X`.

Search for and install the following extensions:

1. **Python** — Microsoft
2. **Jupyter** — Microsoft

The Python extension provides language support and Python environment selection, while the Jupyter extension allows us to execute notebook cells directly within VS Code.

We will use both throughout the example.

---

## Section 3: Setting Up the Project Files

Now that the required software has been installed, we can organize our project.

### 3.1 Opening the Project Folder

In VS Code:

1. Select **File → Open Folder**.
2. Navigate to the folder created in Section 1.
3. Select `MNIST_NeuralNet_Example_last-name-here`.
4. Click **Select Folder**.

Your project folder should now appear in the VS Code Explorer on the left side of the window.

### 3.2 Creating the File Structure

We will keep the project intentionally simple.

Our neural network will be organized into two Python files and one Jupyter notebook. Separating these components makes the code easier to read, modify, and reuse.

Create the following structure:

```text
MNIST_NeuralNet_Example_last-name-here/
│
├── data/
│   └── [MNIST dataset file]
│
├── model.py
├── parameters.py
├── MNIST_Example.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

The files serve the following purposes:

| File | Description |
|---|---|
| `data/` | Contains the small MNIST dataset. |
| `model.py` | Defines the neural network architecture using PyTorch. |
| `parameters.py` | Stores the model and training parameters. |
| `MNIST_Example.ipynb` | Controls data loading, training, and evaluation, allowing us to execute code cell by cell. |
| `requirements.txt` | Lists the Python packages required to reproduce the example. |
| `.gitignore` | Identifies local files that should not be tracked by Git. |
| `README.md` | Provides instructions and documentation for the repository. |

### 3.3 Why Separate the Files?

Instead of writing all our code in a single notebook, we separate the neural network definition from the parameters and training procedure.

For example, `model.py` will contain the one-hidden-layer neural network introduced in the lecture:

\[
h = g(W_1^\top x+\beta_1),
\]

\[
\hat y = W_2^\top h+\beta_2.
\]

The notebook will import this model, load the MNIST observations, and perform training using mini-batches.

This organization allows us to modify the network architecture or training parameters without rewriting the entire notebook.

---

## Section 4: Installing the Required Python Packages

Although Python and VS Code are installed, we still need several Python packages to construct and train our neural network.

We will use PyTorch for neural network computation and Jupyter for interactive execution.

### 4.1 Opening the VS Code Terminal

In VS Code, select:

**Terminal → New Terminal**

On Windows, the terminal will typically open PowerShell.

You can also use the keyboard shortcut:

`Ctrl + Shift + Backtick`

The terminal should open at the bottom of VS Code.

Make sure the terminal is operating within your project folder.

### 4.2 Creating a Python Environment

We will create a local Python environment so that the packages installed for this example are kept separate from other Python projects.

Enter:

```bash
python -m venv .venv
```

On Windows PowerShell, activate the environment using:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, use:

```bash
source .venv/bin/activate
```

If PowerShell prevents activation, the environment can still be used by selecting its Python interpreter in VS Code.

### 4.3 Installing Python Packages

For this example, we will need:

- `torch` — neural network construction, automatic differentiation, and optimization.
- `numpy` — numerical operations and array manipulation.
- `pandas` — tabular data handling, if needed for the supplied MNIST file.
- `matplotlib` — displaying handwritten digits and training results.
- `jupyter` — executing interactive notebooks.
- `ipykernel` — connecting the Python environment to VS Code notebooks.

Create a file named `requirements.txt` and enter:

```text
torch
numpy
pandas
matplotlib
jupyter
ipykernel
```

Then install the packages using:

```bash
python -m pip install -r requirements.txt
```

These instructions assume a standard CPU installation of PyTorch. If installation differs for your system, consult the official instructions at https://pytorch.org/get-started/locally/.

### 4.4 Selecting the Python Environment

Once installation is complete:

1. Open `MNIST_Example.ipynb` in VS Code.
2. Click **Select Kernel** in the upper-right corner.
3. Select the Python environment associated with `.venv`.

This ensures that the notebook uses the packages installed for our project.

---

## Section 5: Running the MNIST Neural Network

We are now ready to connect the mathematical concepts from the lecture to a working neural network.

Our example will use a fully connected neural network with one hidden layer and a ReLU activation function, consistent with the architecture introduced in the Beamer presentation.

The notebook will guide us through the following sequence:

1. **Load the MNIST data.** Read the observations and prepare the inputs and digit labels.
2. **Initialize the neural network.** Construct the model using the architecture defined in `model.py`.
3. **Create mini-batches.** Divide the training observations into smaller groups for parameter updates.
4. **Train the network.** Compute predictions, evaluate the loss, perform backpropagation, and update the parameters.
5. **Evaluate predictions.** Examine how well the trained network predicts handwritten digits.
6. **Save the results.** Store the relevant model outputs for future reference.

During training, we will focus on the following PyTorch operations:

```python
optimizer.zero_grad()

y_hat = model(x_batch)
loss = loss_fn(y_hat, y_batch)

loss.backward()
optimizer.step()
```

These operations correspond directly to the forward pass, loss calculation, backpropagation, and parameter updates discussed in the lecture.

The complete example will be executed through `MNIST_Example.ipynb`, allowing students to examine each stage individually.

---

## Section 6: Saving Your Work with Git and GitHub

Once we have successfully trained our neural network, we will briefly demonstrate how to save the project using Git and GitHub.

Git tracks changes to files locally, while GitHub provides an online location where repositories can be stored and shared.

### 6.1 Initializing a Git Repository

Open the VS Code terminal inside your project folder and enter:

```bash
git init
```

This initializes a local Git repository.

Before adding files, make sure `.gitignore` contains:

```text
.venv/
__pycache__/
.ipynb_checkpoints/
```

We will also exclude any dataset or generated files that should not be uploaded to GitHub.

### 6.2 Saving Your First Commit

Enter:

```bash
git add .
git commit -m "Initial MNIST neural network example"
```

If Git requests your name and email address, configure them using the instructions displayed in the terminal.

The commit records the current state of your project.

### 6.3 Creating a GitHub Repository

1. Visit https://github.com/
2. Sign in or create an account.
3. Select **New repository**.
4. Give the repository a name, such as `MNIST_NeuralNet_Example`.
5. Create the repository without adding a second README or `.gitignore`.

GitHub will provide a repository URL that we can use to connect our local project.

### 6.4 Uploading the Project

In the VS Code terminal, enter:

```bash
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/MNIST_NeuralNet_Example.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and use the actual URL of your new repository.

You may be prompted to authenticate with GitHub.

Once the upload is complete, your project files should be visible in your online repository.

### 6.5 Reproducing the Example on Another Computer

After the repository is published, another student can obtain the code using:

```bash
git clone https://github.com/YOUR-USERNAME/MNIST_NeuralNet_Example.git
cd MNIST_NeuralNet_Example
```

They can then:

1. Open the downloaded folder in VS Code.
2. Create and select a Python environment as described in Section 4.
3. Install the packages using `requirements.txt`.
4. Place the MNIST dataset in the `data/` folder, if it is not included.
5. Open `MNIST_Example.ipynb`.
6. Select the correct Python kernel and run the notebook cells in order.

Following these steps should reproduce the training procedure and allow students to obtain their own results.

---

## References and Additional Resources

- [Python documentation](https://docs.python.org/3/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [Jupyter documentation](https://docs.jupyter.org/)
- [VS Code Python documentation](https://code.visualstudio.com/docs/python/python-tutorial)
- [Git documentation](https://git-scm.com/doc)
- [GitHub documentation](https://docs.github.com/)

**Lecture:** *A Gentle Introduction to Neural Networks for Predictive Modeling*, Owen Visser, University of Florida, 2026.