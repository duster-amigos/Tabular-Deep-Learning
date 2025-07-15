import torch
from loguru import logger

def get_device():
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info('Using CUDA GPU.')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info('Using Apple MPS GPU.')
        else:
            device = torch.device('cpu')
            logger.info('Using CPU.')
        return device
    except Exception as e:
        logger.error(f"Error determining device: {e}")
        raise 