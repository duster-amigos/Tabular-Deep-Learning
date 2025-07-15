import torch
import numpy as np
from loguru import logger
from tabtransformer.model import TabTransformer
from utils.device import get_device
import argparse
import os


def load_data(path, dtype=None):
    try:
        ext = os.path.splitext(path)[1]
        if ext == '.npy':
            arr = np.load(path)
        elif ext == '.csv':
            arr = np.loadtxt(path, delimiter=',', dtype=dtype)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        logger.info(f"Loaded data from {path} with shape {arr.shape}")
        return arr
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise

def main():
    try:
        parser = argparse.ArgumentParser(description='Train TabTransformer on tabular data')
        parser.add_argument('--categ_path', type=str, required=True, help='Path to categorical features file (.npy or .csv)')
        parser.add_argument('--cont_path', type=str, required=True, help='Path to continuous features file (.npy or .csv)')
        parser.add_argument('--labels_path', type=str, required=True, help='Path to labels file (.npy or .csv)')
        parser.add_argument('--categories', type=int, nargs='+', required=True, help='Unique values per categorical column')
        parser.add_argument('--num_continuous', type=int, required=True, help='Number of continuous columns')
        parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes')
        parser.add_argument('--dim', type=int, default=32)
        parser.add_argument('--depth', type=int, default=6)
        parser.add_argument('--heads', type=int, default=8)
        parser.add_argument('--attn_dropout', type=float, default=0.1)
        parser.add_argument('--ff_dropout', type=float, default=0.1)
        parser.add_argument('--mlp_hidden_mults', type=int, nargs='+', default=[4, 2])
        parser.add_argument('--mlp_act', type=str, default='relu', choices=['relu', 'selu'])
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--lr', type=float, default=1e-3)
        args = parser.parse_args()

        device = get_device()
        logger.info(f"Training on device: {device}")

        x_categ = load_data(args.categ_path, dtype=np.int64)
        x_cont = load_data(args.cont_path, dtype=np.float32)
        y = load_data(args.labels_path, dtype=np.int64)

        assert x_categ.shape[0] == x_cont.shape[0] == y.shape[0], "Mismatched number of samples"
        assert x_categ.shape[1] == len(args.categories), "Mismatched number of categorical columns"
        assert x_cont.shape[1] == args.num_continuous, "Mismatched number of continuous columns"

        mlp_act = torch.nn.ReLU() if args.mlp_act == 'relu' else torch.nn.SELU()
        model = TabTransformer(
            categories=tuple(args.categories),
            num_continuous=args.num_continuous,
            dim=args.dim,
            dim_out=args.num_classes,
            depth=args.depth,
            heads=args.heads,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            mlp_hidden_mults=tuple(args.mlp_hidden_mults),
            mlp_act=mlp_act
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        n_samples = x_categ.shape[0]
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            n_batches = 0
            for idx in range(0, n_samples, args.batch_size):
                xb_categ = torch.tensor(x_categ[idx:idx+args.batch_size], dtype=torch.long, device=device)
                xb_cont = torch.tensor(x_cont[idx:idx+args.batch_size], dtype=torch.float32, device=device)
                yb = torch.tensor(y[idx:idx+args.batch_size], dtype=torch.long, device=device)
                optimizer.zero_grad()
                out = model(xb_categ, xb_cont)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/n_batches:.4f}")
    except Exception as e:
        logger.error(f"Error in training script: {e}")
        raise

if __name__ == '__main__':
    main() 