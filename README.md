# Non-negative similarity matching in PyTorch

This is an implementation of non-negative similarity matching (NSM) for PyTorch focusing on ease of use, extensibility, and speed.

## Table of Contents

- [Installation](#installation)
- [Example usage](#examples)
- [Features](#features)
- [Questions?](#questions)

## Installation<a name="installation"></a>

It is strongly recommended to use a virtual environment when working with this code. The installation instructions below include the commands for creating the virtual environment, using either `conda` (recommended) or `venv`.

### Using `conda`

If you do not have `conda` installed, the easiest way to get started is with [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Follow the installation instructions for your system.

Next, create a new environment and install the necessary pre-requisites using

```sh
conda env create -f environment.yml
```

Finally, install the `pynsm` package:

```sh
pip install -e .
```

The `-e` marks this as an "editable" install — this means that changes made to the code will automatically take effect without having to reinstall the package. This is mainly useful for developers.

### Using `venv`

**TODO:** explain how not to use the system Python.

Create a new virtual environment by running the following command in a terminal inside the main folder of the repository:

```sh
python -m venv env
```

This creates a subfolder called `env` containing the files for the virtual environment. Next we need to activate the environment and install the necessary pre-requisites

```sh
source env/bin/activate
pip install -r requirements.txt
```

Finally, install the `pynsm` package:

```sh
pip install -e .
```

The `-e` marks this as an "editable" install — this means that changes made to the code will automatically take effect without having to reinstall the package. This is mainly useful for developers.

## Example Usage<a name="examples"></a>

See the notebooks in the [`examples`](examples) folder to get started with the package.

## Features<a name="features"></a>

### Neural Similarity Model (NSM) Convolution

The code defines a neural network model called `NSM_Conv`. This model consists of an encoder and a competitor. The encoder is implemented as a convolutional layer and the competitor is implemented as a linear layer. The `forward` method performs the forward pass of the model. The model also includes methods for pooling the output and calculating the loss.

### ZCA Whitening

The code includes a ZCA whitening function `computeZCAMatrix` that computes the ZCA matrix for a set of input observations `X`. The function performs normalization, reshaping, covariance computation, singular value decomposition (SVD), and builds the ZCA matrix. The whitening transformation is implemented in the `ZCATransformation` class, which takes the ZCA matrix and transformation mean as inputs and applies the transformation to a given tensor image.

### Training the Model

The model is trained using the `train` method of the `NSM_Conv` class. The code defines an instance of the `NSM_Conv` model and initializes the optimizer. It then iterates over the training data and performs forward and backward passes to update the model parameters.

### Embedding Extraction

After training the model, the code extracts embeddings for both the training and test datasets. The `train_loader` and `test_loader` iterate over the training and test datasets, respectively. For each batch of data, the model is used to compute embeddings, and the embeddings are appended to the `train_embedded_data` and `test_embedded_data` lists. The corresponding labels are also stored in `train_labels_data` and `test_labels_data`, respectively.

### Classification

The code performs classification on the extracted embeddings using the `SGDClassifier` from the `sklearn` library. The embeddings are reshaped, and a random subset of data points is selected for training. The classifier is trained on the selected data using the `fit` method. The accuracy of the trained classifier is evaluated on the test data using the `score` method.

### Supervised NSM

The second part of the code introduces the `Supervised_NSM_Conv` class, which implements supervised learning. It includes additional functionality for handling labeled data during training.

- There is an additional encoder for the labels, `encoder_labels`, which processes the label information.
- The forward pass includes the labels as input and incorporates them into the computation.
- The loss function considers the label information in addition to the data and updates the loss accordingly.
- The training method also incorporates the labels during backpropagation and weight updates.

The `Supervised_NSM_Conv` class also includes additional methods for pooling the output and visualizing the learned features.

## Questions?<a name="features"></a>

Please contact us by opening an issue on GitHub.
