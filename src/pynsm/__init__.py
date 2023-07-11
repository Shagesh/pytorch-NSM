__version__ = "0.9.0"
__author__ = "Shagesh Sridharan, Tiberiu Tesileanu, Yanis Bahroun"
__email__ = "shagesh1996@gmail.com,ttesileanu@gmail.com,ybahroun@flatironinstitute.org"
__uri__ = "https://github.com/Shagesh/pytorch-NSM"
__description__ = "A PyTorch implementation of non-negative similarity matching"
__license__ = "MIT"

from .arch import IterationModule, IterationLossModule
from .arch import SimilarityMatching, MultiSimilarityMatching
from .arch import SupervisedSimilarityMatching
from .util import extract_embeddings

__all__ = [
    "IterationModule",
    "IterationLossModule",
    "SimilarityMatching",
    "MultiSimilarityMatching",
    "SupervisedSimilarityMatching",
    "extract_embeddings",
    "__version__",
]
