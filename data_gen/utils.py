import numpy as np
import random
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sample_random(X_all, N):
    set_seed(0)
    idx_all = []
    for i in range(X_all.shape[0]):
        idx_all.append(i)

    idx = np.random.choice(X_all.shape[0], N, replace=False)
    idx_complement = list(set(idx_all) - set(idx))
    idx_test = np.random.choice(idx_complement, 1000, replace=False)
    
    idx_val_complement = list(set(idx_complement) - set(idx_test))
    idx_val = np.random.choice(idx_val_complement, 1000, replace=False)

    X_sampled = X_all[idx, :]
    # X_test = X_all[idx_test, :]
    return X_sampled, idx_test, idx_val

def set_activation(activation):
    if activation == 'identity':
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return  nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        print("WARNING: unknown activation function!")
        return -1
