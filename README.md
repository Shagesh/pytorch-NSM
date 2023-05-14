# pytorch-NSM

# Code Documentation

This is the documentation page for the code. It provides an overview of the code, its functionalities, and the purpose of each component. The code is divided into several sections, each serving a specific purpose.

## Table of Contents
- [Dependencies](#dependencies)
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [ZCA Whitening](#zca-whitening)
- [Neural Similarity Model (NSM) Convolution](#neural-similarity-model-nsm-convolution)
- [Training the Model](#training-the-model)
- [Embedding Extraction](#embedding-extraction)
- [Classification](#classification)

## Dependencies<a name="dependencies"></a>
The following external libraries are imported in the code:
- `time`: Provides various time-related functions.
- `tqdm`: A library for creating progress bars in the command line.
- `scipy`: A library for scientific computing in Python.
- `numpy`: A library for numerical operations in Python.
- `pandas`: A library for data manipulation and analysis.
- `gc`: A module for garbage collection in Python.
- `sklearn`: A library for machine learning algorithms and tools.
- `re`: A module for regular expression operations in Python.
- `einops`: A library for tensor operations.
- `matplotlib`: A library for data visualization in Python.
- `torch`: The PyTorch library for deep learning.
- `torchvision`: A PyTorch library for vision tasks.
- `keras`: A high-level neural networks API written in Python.

## Data Loading and Preprocessing<a name="data-loading-and-preprocessing"></a>
This section includes code for loading and preprocessing the input data. It supports two datasets: MNIST and CIFAR10. The data is loaded using the respective `datasets` modules from `torchvision`. The input images are normalized and converted to tensors using `transforms.ToTensor()`. The training and test datasets are created using `datasets.MNIST` or `datasets.CIFAR10` depending on the chosen dataset. The training and test data loaders are created using `torch.utils.data.DataLoader`.

## ZCA Whitening<a name="zca-whitening"></a>
The code includes a ZCA whitening function `computeZCAMAtrix` that computes the ZCA matrix for a set of input observations `X`. The function performs normalization, reshaping, covariance computation, singular value decomposition (SVD), and builds the ZCA matrix. The whitening transformation is implemented in the `ZCATransformation` class, which takes the ZCA matrix and transformation mean as inputs and applies the transformation to a given tensor image.

## Neural Similarity Model (NSM) Convolution<a name="neural-similarity-model-nsm-convolution"></a>
The code defines a neural network model called NSM_Conv. This model consists of an encoder and a competitor. The encoder is implemented as a convolutional layer, and the competitor is implemented as a linear layer. The `forward` method performs the forward pass of the model. The model also includes methods for pooling the output and calculating the loss.

## Training the Model<a name="training-the-model"></a>
The model is trained using the `train` method of the NSM_Conv class. The code defines an instance of the NSM_Conv model and initializes the optimizer. It then iterates over the training data and performs forward and backward passes to update the model parameters.

## Embedding Extraction<a name="embedding-extraction"></a>
After training the model, the code extracts embeddings for both the training and test datasets. The `train_loader` and `test_loader` iterate over the training and test datasets, respectively. For each batch of data, the model is used to compute embeddings, and the embeddings are appended to the `train_embedded_data` and `test_embedded_data` lists. The corresponding labels are also stored in `train_labels_data` and `test_labels_data`, respectively.

## Classification<a name="classification"></a>
The code performs classification on the extracted embeddings using the `SGDClassifier` from the `sklearn` library. The embeddings are reshaped, and a random subset of data points is selected for training. The classifier is trained on the selected data using the `fit` method. The accuracy of the trained classifier is evaluated on the test data using the `score` method.

## Example Usage
To use this code, follow these steps:

1. Install the required dependencies mentioned in the code documentation.
2. Import the necessary libraries and modules.
3. Load and preprocess the desired dataset (MNIST or CIFAR10).
4. Define and train the NSM_Conv model.
5. Extract embeddings for the training and test datasets.
6. Perform classification on the extracted embeddings.

Remember to ensure that the necessary data files are available and properly formatted.

Please note that this is just an overview of the code functionality. For detailed explanations of each function and class, please refer to the code comments and documentation.

## Part 2: Supervised NSM_Conv

The second part of the code introduces the `Supervised_NSM_Conv` class, which extends the `NSM_Conv` class to support supervised learning. It includes additional functionality for handling labeled data during training.

### Class: Supervised_NSM_Conv
This class inherits from the `NSM_Conv` class and adds the following functionality:
- It includes an additional encoder for the labels, `encoder_labels`, which processes the label information.
- The forward pass includes the labels as input and incorporates them into the computation.
- The loss function considers the label information in addition to the data and updates the loss accordingly.
- The training method also incorporates the labels during backpropagation and weight updates.

The `Supervised_NSM_Conv` class also includes additional methods for pooling the output and visualizing the learned features.

### Training and Embedding Extraction
The code proceeds to train the `Supervised_NSM_Conv` model using the training data. The training process is similar to the previous part, but now the model is trained with both the images and the corresponding labels. The loss function is modified to incorporate label information.

After training, the code extracts embeddings for both the training and test datasets using the trained model. The process is similar to the previous part, but now the model takes into account the labels when computing the embeddings.

### Classification
Finally, the code performs classification on the extracted embeddings using the `SGDClassifier` from the `sklearn` library. The embeddings are reshaped, and a random subset of data points is selected for training. The classifier is trained on the selected data using the `fit` method. The accuracy of the trained classifier is evaluated on the test data using the `score` method.

### Visualization
The code also includes a visualization step where the learned filters are displayed. The filters are extracted from the `encoder` of the trained model and visualized using matplotlib.

## Example Usage
To use this code, follow these steps:

1. Install the required dependencies mentioned in the code documentation.
2. Import the necessary libraries and modules.
3. Load and preprocess the desired dataset (MNIST or CIFAR10).
4. Define and train the `Supervised_NSM_Conv` model.
5. Extract embeddings for the training and test datasets using the trained model.
6. Perform classification on the extracted embeddings.
7. Visualize the learned filters.

Remember to ensure that the necessary data files are available and properly formatted.

Please note that this is just an overview of the code functionality. For detailed explanations of each function and class, please refer to the code comments and documentation.

If you have any further questions, feel free to ask!
