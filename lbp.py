from skimage.feature import local_binary_pattern
from typing import Literal
import skimage
import numpy as np

def lbp(image : np.ndarray,
        P : int = 24,
        R : int = 4,
        method : Literal['default', 'ror', 'uniform', 'nri_uniform', 'var'] = 'uniform'):

    assert isinstance(image, np.ndarray) and len(image.shape) == 2
    desc = local_binary_pattern(image, P, R, method=method)
    n_bins = int(desc.max() + 1)
    hist, _ = np.histogram(desc, density=True, bins=n_bins, range=(0, n_bins))

    return hist