import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import FashionMNIST

SEED = 42

def make_splits():
    ds = FashionMNIST(root='./data', train=True, download=True)
    y = ds.targets.numpy()
    idx = np.arange(60000)
    
    train_idx, val_idx = train_test_split(
        idx, test_size=10000, stratify=y, random_state=SEED
    )
    train_10k_idx, _ = train_test_split(
        train_idx, train_size=10000, stratify=y[train_idx], 
        random_state=SEED+1
    )
    np.savez('splits/splits.npz', 
             train_idx=train_idx, 
             val_idx=val_idx, 
             train_10k_idx=train_10k_idx,
             seed=SEED)
    print("Saved splits/splits.npz")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Train10k: {len(train_10k_idx)}")

if __name__ == "__main__":
    make_splits()
