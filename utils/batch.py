import numpy as np
import torch
from loguru import logger

def batch_iter(data, batch_size, shuffle=True):
    try:
        n = len(data)
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, n, batch_size):
            excerpt = indices[start_idx:start_idx + batch_size]
            if isinstance(data, np.ndarray):
                batch = data[excerpt]
            elif torch.is_tensor(data):
                batch = data[excerpt]
            else:
                raise TypeError("Data must be numpy array or torch tensor.")
            logger.debug(f"Yielding batch: start={start_idx}, size={len(batch)}")
            yield batch
    except Exception as e:
        logger.error(f"Error in batch_iter: {e}")
        raise 