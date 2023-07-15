"""Define a class to perform a ZCA transform."""

import numpy as np
import torch
from torch.utils.data import Dataset


def computeZCAMAtrix(X):
    # This function computes the ZCA matrix for a set of observables X where
    # rows are the observations and columns are the variables (M x C x W x H matrix)
    # C is number of color channels and W x H is width and height of each image

    X = np.asarray(X)

    # compute mean and std and normalize the data to -1 1 range with 1 std
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_std = (X - mean) / std

    # reshape data from M x C x W x H to M x N where N=C x W x H
    X_std_flat = X_std.reshape(len(X_std), -1)

    # compute the covariance
    cov = np.cov(X_std_flat, rowvar=False)  # cov is (N, N)

    # singular value decomposition
    U, S, V = np.linalg.svd(cov)  # U is (N, N), S is (N,1) V is (N,N)
    # build the ZCA matrix which is (N,N)
    epsilon = 1e-5
    zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))

    return (torch.from_numpy(zca_matrix).float(), mean, std)


class ZCATransformation(object):
    def __init__(self, transformation_matrix, transformation_mean):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError(
                "transformation_matrix should be square. Got "
                + "[{} x {}] rectangular matrix.".format(*transformation_matrix.size())
            )
        self.transformation_matrix = transformation_matrix
        self.transformation_mean = transformation_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (N, C, H, W) to be whitened.
        Returns:
            Tensor: Transformed image.
        """
        count = tensor.size(1) * tensor.size(2) * tensor.size(3)
        if count != self.transformation_matrix.size(0):
            raise ValueError(
                "tensor and transformation matrix have incompatible shape."
                + "[{} x {} x {}] != ".format(*tensor[0].size())
                + "{}".format(self.transformation_matrix.size(0))
            )
        batch = tensor.size(0)

        flat_tensor = tensor.view(batch, -1)
        transformed_tensor = torch.mm(
            flat_tensor - self.transformation_mean, self.transformation_matrix
        )

        tensor = transformed_tensor.view(tensor.size())
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += str(self.transformation_matrix.numpy().tolist()) + ")"
        return format_string
