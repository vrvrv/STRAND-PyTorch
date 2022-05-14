import math
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm

from .functions import *

def get_loss_fn(objective='poisson', tau=50):
    if objective == 'nbconst':
        def loss_fn(C, Chat):
            _Lij = tau \
                   * math.log(tau) \
                   - math.lgamma(tau) \
                   + torch.lgamma(C + tau) \
                   + C * torch.log(Chat) \
                   - torch.log(Chat + tau) \
                   * (tau + C) \
                   - torch.lgamma(C + 1)

            return - _Lij

    elif objective == 'poisson':
        def loss_fn(C, Chat):
            _Lij = C \
                   * torch.log(Chat) \
                   - Chat \
                   - torch.lgamma(C + 1)
            return - _Lij
    else:
        raise NotImplementedError

    return loss_fn
