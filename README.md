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


### 1.2 Jupyter Notebooks

Jupyter notebooks allow us to execute Python code in individual cells, inspect intermediate calculations, and display results without running an entire Python script.

We will use Jupyter notebooks directly within VS Code, so a separate Jupyter desktop application is not required.

We will install the necessary Python packages in Section 4.

For additional information, visit https://jupyter.org/.

### 1.3 Installing Git

Git is a version control system that allows us to track changes to our code and save our work in repositories.

We will use Git to download the example repository directly to our computers.

1. Visit https://git-scm.com/downloads
2. Download Git for your operating system.
3. Follow the installation instructions.

To verify the installation, enter:

```bash
git --version
```

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

## Section 3: Downloading the GitHub Repository

Now that the necessary software has been installed, we can download the example repository.

The repository already contains the Python scripts, Jupyter notebook, and MNIST dataset needed for the example. You do not need to create these files yourself.

### 3.1 Opening the VS Code Terminal

Open VS Code and select:

**Terminal → New Terminal**

On Windows, this will typically open a PowerShell terminal.

You can also use the keyboard shortcut:

`Ctrl + Shift + Backtick`

### 3.2 Cloning the Repository

First, navigate to the location where you would like to save the project. For example, on Windows:

```powershell
cd "$HOME\Documents"
```

Next, clone the repository from GitHub:

```powershell
git clone https://github.com/owenvisser/PHC-6099-Neural-Network-Walkthrough.git
```

Git will download the project and create a folder containing all the necessary files.

### 3.3 Opening the Project in VS Code

After cloning the repository, enter:

```powershell
cd PHC-6099-Neural-Network-Walkthrough
code .
```

Alternatively, select **File → Open Folder** in VS Code and navigate to the downloaded repository.

### 3.4 Understanding the File Structure

Our project is intentionally organized into a small number of files:

```text
MNIST_NeuralNet_Example/
│
├── data/
│   └── mnist.pkl.gz
│
├── model.py
├── parameters.py
├── MNIST_Example.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

| File | Description |
|---|---|
| `data/mnist.pkl.gz` | Contains the MNIST observations and digit labels used for training and evaluation. |
| `model.py` | Defines the neural network architecture using PyTorch. |
| `parameters.py` | Stores the model and training parameters. |
| `MNIST_Example.ipynb` | Controls data loading, model training, and evaluation. |
| `requirements.txt` | Lists the Python packages required to run the example. |
| `.gitignore` | Specifies files and folders that should not be tracked by Git. |
| `README.md` | Provides instructions for downloading and running the project. |

The model architecture and parameters are kept separate from the Jupyter notebook so that we can examine each part of the training procedure independently.

**Note:** The `results/` folder is intentionally not included in the downloaded repository. It will be created when you execute the notebook and save the results of your own neural network.

---

## Section 4: Installing the Required Python Packages

Although the repository contains the necessary code and data, we still need to install the Python packages required to execute the example.

### 4.1 Creating a Python Environment

In the VS Code terminal, make sure you are inside the downloaded project folder.

Create a Python environment by entering:

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

If PowerShell prevents activation, you can still use the environment by selecting its Python interpreter in VS Code.

### 4.2 Installing the Packages

We will use the following Python packages:

- `torch` — neural network construction, automatic differentiation, and optimization.
- `numpy` — numerical operations and array manipulation.
- `pandas` — tabular data handling, where needed.
- `matplotlib` — displaying handwritten digits and training results.
- `jupyter` — executing interactive notebooks.
- `ipykernel` — connecting the Python environment to VS Code notebooks.

The repository already contains a `requirements.txt` file listing the necessary packages.

To install them, enter:

```bash
python -m pip install -r requirements.txt
```

This command installs the packages into your active Python environment.

These instructions assume a standard CPU installation of PyTorch. However, if you are certain that your computer has a compatible NVIDIA GPU, you may install the CUDA-enabled version of PyTorch to accelerate model training.

First, verify that your NVIDIA GPU and drivers support CUDA by entering the following command in the VS Code terminal:

```powershell
nvidia-smi
```

If your GPU and drivers support CUDA 13.0, you can install the corresponding CUDA-enabled version of PyTorch by entering:

```powershell
python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130
```

For other CUDA versions or platform-specific installation instructions, visit https://pytorch.org/get-started/locally/ and select the appropriate configuration.

Once installation is complete, verify that PyTorch can access your GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If the command returns `True`, PyTorch can access your GPU. Our Jupyter notebook will automatically use CUDA when available; otherwise, it will run on the CPU.

### 4.3 Selecting the Python Environment

Once installation is complete:

1. Open `MNIST_Example.ipynb` in VS Code.
2. Click **Select Kernel** in the upper-right corner.
3. Select the **Python Environment** associated with our path, specifically `.venv\Scripts\python.exe`.

This ensures that the notebook uses the packages installed for our project.

---

## Section 5: Running the MNIST Neural Network

We are now ready to connect the mathematical concepts from the lecture to a working neural network.

Our example uses a fully connected neural network with one hidden layer and a ReLU activation function, consistent with the architecture introduced in the lecture.

### 5.1 Opening the Notebook

In the VS Code Explorer, open:

```text
MNIST_Example.ipynb
```

The notebook contains the code needed to load the MNIST dataset, initialize the model, train the neural network, and evaluate its predictions.

We will execute the notebook one cell at a time to better understand each step.

### 5.2 Understanding the Training Procedure

The notebook is organized around the following operations:

1. **Loading the data:** Read the MNIST observations and prepare the inputs and digit labels.
2. **Initializing the network:** Construct the one-hidden-layer model defined in `model.py`.
3. **Creating mini-batches:** Divide the training observations into smaller groups.
4. **Training the network:** Calculate predictions, evaluate the loss, and update the parameters.
5. **Evaluating predictions:** Examine how well the trained network predicts handwritten digits.
6. **Saving the results:** Store the relevant outputs from the training procedure.

The central training procedure follows the structure:

```python
optimizer.zero_grad()

y_hat = model(x_batch)
loss = loss_fn(y_hat, y_batch)

loss.backward()
optimizer.step()
```

These operations correspond directly to the concepts introduced in the lecture:

- `optimizer.zero_grad()` clears previously calculated gradients.
- `model(x_batch)` performs the forward pass.
- `loss_fn(...)` evaluates the loss function.
- `loss.backward()` applies backpropagation to calculate gradients.
- `optimizer.step()` updates the model parameters.

### 5.3 Executing the Notebook

To run the example:

1. Begin with the first code cell in `MNIST_Example.ipynb`.
2. Press `Shift + Enter` to execute the current cell and move to the next.
3. Continue executing the cells in order.
4. Examine the outputs as the neural network is trained and evaluated.

Alternatively, you may select **Run All** at the top of the notebook to execute every cell.

Training time will depend on your computer and the model parameters.

### 5.4 Examining Your Results

Once training is complete, examine the outputs produced by the notebook.

Depending on the saved outputs, these may include:

- Training loss across epochs.
- Predicted handwritten digits.
- Comparisons between predicted and observed labels.
- Summaries of predictive performance.

The notebook will save the designated outputs to a folder named:

```text
results/
```

This folder is intentionally excluded from GitHub using `.gitignore`, so each student must execute the notebook to generate their own results.

The saved outputs allow you to inspect the results after training without repeating the entire procedure.

---

## Section 6: Using Git to Save Your Work

Git allows us to track changes to our project and save versions of our code.

Because we cloned an existing repository, Git is already initialized in the downloaded project folder.

### 6.1 Checking Your Repository

Open the VS Code terminal and enter:

```bash
git status
```

This command displays any tracked files that have been modified.

### 6.2 Saving Changes Locally

If you make changes to the Python scripts or notebook, you can record them with Git:

```bash
git add .
git commit -m "Updated MNIST neural network example"
```

This saves a new version of your project locally.

If Git requests your name and email address, follow the instructions displayed in the terminal to configure them.

### 6.3 Uploading Your Own Repository

If you would like to save your modified project online, you can create your own GitHub repository.

Visit https://github.com/ and create a new empty repository.

Then connect your local project to your own repository:

```bash
git remote set-url origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the URL with the address of your new GitHub repository.

The first command changes the remote destination from the classroom repository to your own repository. The second uploads your local `main` branch, including your committed changes.

The `results/` folder and local Python environment will remain excluded as specified in `.gitignore`.

---

## References and Additional Resources

- [Python documentation](https://docs.python.org/3/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [Jupyter documentation](https://docs.jupyter.org/)
- [VS Code Python documentation](https://code.visualstudio.com/docs/python/python-tutorial)
- [Git documentation](https://git-scm.com/doc)
- [GitHub documentation](https://docs.github.com/)

**Lecture:** *A Gentle Introduction to Neural Networks for Predictive Modeling*, Owen Visser, University of Florida, 2026.