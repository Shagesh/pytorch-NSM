# How to build a supervised classifier

Here we will learn how to build, train, and test a digit classifier using supervised similarity matching. Check out the background page on [supervised learning](supervised.md) for the theory behind this use of similarity matching.

## Training
We assume we load MNIST in the usual way:

```python
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from pynsm import SupervisedSimilarityMatching

# load MNIST
train_data = datasets.MNIST(download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128)
```

Building the model requires an encoder, which we choose as a convolutional network in this case:

```python
# build the model
torch.manual_seed(42)

num_kernels = 50
num_labels = 10
encoder = nn.Conv2d(1, num_kernels, 6, stride=1, padding=0, bias=False)
model = SupervisedSimilarityMatching(
    encoder, num_labels, num_kernels, iteration_projection=torch.nn.ReLU()
)
```

To train the model, we use a standard loop:

```python
# train the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(3):
    for images, labels in train_loader:
        labels = F.one_hot(labels, num_classes=num_labels).float()

        outputs = model(images, labels)
        loss = model.loss(images, labels, outputs)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

Note that unlike typical PyTorch models, the `SimilarityMatching` instance itself contains a method for calculating the loss. This choice was made because in these models, the iteration needed in the fast forward iteration, as well as the slower weight updates, are based on the same loss function, and so it makes sense for the model to be aware of it. Note, however, that the user is free to modify the loss (e.g., by adding regularizer terms), or altogether replace it with something else.

## Inference and testing
We used both the input images as well as the labels for training in the loop above. Note, however, that the output of the network is *not* a probability distribution over digit classes, as in usual classifier, but instead it is a joint encoding of both the image and the label, as in self-supervised learning. To use this for classification, we can tack on any classifier, for example, an [SVM](https://en.wikipedia.org/wiki/Support_vector_machine). To accelerate learning and avoid overfitting, we first use max-pooling to reduce the dimensionality of the model output:

```python
inference_model = nn.Sequential(model, nn.MaxPool2d(kernel_size=2, stride=2))
```

Any classifier can be used, but here, as an example, we want to use the output of the inference model to train an SVM using [scikit-learn](https://scikit-learn.org/stable/). The `pynsm` package includes a utility function for extracting the model's outputs for all the samples in the dataset:

```python
train_embed = extract_embeddings(inference_model, train_loader)
```

Each sample here is a matrix, so the samples need to be flattened before passing to the classifier:

```python
from sklearn.linear_model import SGDClassifier

n_train = len(train_embed.output)
n_test = len(test_embed.output)

classifier = SGDClassifier(random_state=1337)

train_embed_flat = train_embed.output.reshape(n_train, -1)
classifier.fit(train_embed_flat, train_embed.label)
print(
    f"Classification accuracy: "
    f"{classifier.score(train_data, train_embed.label) * 100:.2f}%"
)
```
