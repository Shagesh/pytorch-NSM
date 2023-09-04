"""
# Supervised NSM

Demonstrating supervised non-negative similarity matching. In addition, this uses a
convolutional encoder.
"""

# %%
import time
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from sklearn.linear_model import SGDClassifier

from pynsm import SupervisedSimilarityMatching, extract_embeddings

# %% [markdown]
# ## Load dataset and create data loaders
# Using standard `torchvision` calls to load the dataset.

# %%
transform = transforms.ToTensor()

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
classes = [str(i) for i in range(10)]

# %%
# Create training and test data loaders.
batch_size = 128

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)

# %%
# Show sample images to test the loading process.
torch.manual_seed(42)

X_batch, y_batch = next(iter(train_loader))
X_max = X_batch.max()
X_min = X_batch.min()

print(f"batch min={X_min:.3g}, mean={X_batch.mean():.3g}, max={X_max:.3g}")

# create a grid of 3x3 images
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(8, 8))
for i in range(3):
    for j in range(3):
        batch_idx = i * 3 + j
        crt_X = X_batch[batch_idx]
        ax[i][j].imshow(crt_X.numpy().squeeze(), vmin=X_min, vmax=X_max, cmap="gray")
        ax[i][j].set_title(classes[y_batch[batch_idx].item()])

# %% [markdown]
# ## Train supervised convolutional NSM

# %%
torch.manual_seed(42)
n_epochs = 1
num_labels = len(classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}.")

# the encoder layer is convolutional
encoder = nn.Conv2d(1, 50, 6, stride=1, padding=0, bias=False)
model = SupervisedSimilarityMatching(
    encoder, num_labels, 50, label_bias=False, iteration_projection=torch.nn.ReLU()
).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

t0 = time.time()
running_loss = []
slow_tqdm = lambda *args, **kwargs: tqdm(*args, mininterval=30, **kwargs)
for epoch in range(n_epochs):
    pbar = slow_tqdm(train_loader, desc=f"epoch {epoch + 1} / {n_epochs}")
    sample = 0
    for idx, data in enumerate(pbar):
        images, labels = data

        images = images.to(device)
        labels = F.one_hot(labels, num_classes=num_labels).to(device).float()

        outputs = model(images, labels)
        loss = model.loss(images, labels, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

        pbar.set_postfix({"sample": sample, "loss": running_loss[-1]}, refresh=False)

        sample += len(images)

t1 = time.time()
print(f"Training took {t1 - t0:.2f} seconds.")

# %%
# Show learning curve.
fig, ax = plt.subplots()
ax.plot(running_loss, lw=1.0)
ax.set_xlabel("batch")
ax.set_ylabel("loss")
sns.despine(ax=ax, offset=10)

# %% [markdown]
# Showcase some of the convolutional filters.

# %%
filters = model.encoders[0].weight.detach().cpu().numpy()  # type: ignore

fig, ax = plt.subplots(7, 7, sharex=True, sharey=True, figsize=(8, 8))
for i in range(7):
    for j in range(7):
        crt_filter = filters[i * 7 + j, 0]
        crt_max = np.max(np.abs(crt_filter))
        ax[i][j].imshow(crt_filter, vmin=-crt_max, vmax=crt_max, cmap="RdBu")

# %% [markdown]
# ## Test how well the pre-trained network can help with classification

# %% [markdown]
# We add a max pooling operation to the output from our convolutional NSM module, then
# check how well an SVM trained on this final output manages to classify digits.

# %%
inference_model = nn.Sequential(model, nn.MaxPool2d(kernel_size=2, stride=2))
t0 = time.time()
train_embed = extract_embeddings(inference_model, train_loader, progress=slow_tqdm)
t1 = time.time()
print(f"Embedding training set took {t1 - t0:.2f} seconds.")

t0 = time.time()
test_embed = extract_embeddings(inference_model, test_loader, progress=slow_tqdm)
t1 = time.time()
print(f"Embedding test set took {t1 - t0:.2f} seconds.")

# %%
# We use `scikit-learn` to fit an SVM to the embedded images.
n_train = len(train_embed.output)
n_test = len(test_embed.output)

classifier = SGDClassifier(random_state=123)

train_data = train_embed.output.reshape(n_train, -1)
classifier.fit(train_data, train_embed.label)
train_error = classifier.score(train_data, train_embed.label)
print(
    f"Accuracy on {len(train_embed.output)} training images: {100 * train_error:.1f}%."
)

test_error = classifier.score(test_embed.output.reshape(n_test, -1), test_embed.label)
print(f"Accuracy on {len(test_embed.output)} test images: {100 * test_error:.1f}%.")

# %%
