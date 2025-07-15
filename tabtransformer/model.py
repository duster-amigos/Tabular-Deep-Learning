import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class ColumnEmbedding(nn.Module):
    """
    Embeds each categorical column with a unique column identifier and value embedding, as in TabTransformer.
    """
    def __init__(self, categories, dim):
        super().__init__()
        try:
            self.categories = categories
            self.num_columns = len(categories)
            self.dim = dim
            self.col_id_dim = max(1, dim // 8)
            self.val_emb_dim = dim - self.col_id_dim
            self.col_ids = nn.Parameter(torch.randn(self.num_columns, self.col_id_dim))
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_cat + 1, self.val_emb_dim)  # +1 for missing value
                for num_cat in categories
            ])
            logger.info(f"Initialized ColumnEmbedding: {self.num_columns} columns, dim={dim}")
        except Exception as e:
            logger.error(f"Error initializing ColumnEmbedding: {e}")
            raise

    def forward(self, x_categ):
        try:
            # x_categ: (batch, num_columns)
            batch_size = x_categ.size(0)
            outs = []
            for i, emb in enumerate(self.embeddings):
                val_emb = emb(x_categ[:, i])  # (batch, val_emb_dim)
                col_id = self.col_ids[i].unsqueeze(0).expand(batch_size, -1)  # (batch, col_id_dim)
                out = torch.cat([col_id, val_emb], dim=-1)  # (batch, dim)
                outs.append(out)
            result = torch.stack(outs, dim=1)  # (batch, num_columns, dim)
            logger.debug(f"ColumnEmbedding output shape: {result.shape}")
            return result
        except Exception as e:
            logger.error(f"Error in ColumnEmbedding forward: {e}")
            raise

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, attn_dropout, ff_dropout):
        super().__init__()
        try:
            self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(dim)
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(ff_dropout)
            )
            self.norm2 = nn.LayerNorm(dim)
            logger.info(f"Initialized TransformerBlock: dim={dim}, heads={heads}")
        except Exception as e:
            logger.error(f"Error initializing TransformerBlock: {e}")
            raise

    def forward(self, x):
        try:
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
            logger.debug(f"TransformerBlock output shape: {x.shape}")
            return x
        except Exception as e:
            logger.error(f"Error in TransformerBlock forward: {e}")
            raise

class TabTransformer(nn.Module):
    """
    TabTransformer for numerical tabular data, following the original paper.
    Includes robust logging and error handling.
    """
    def __init__(self, categories, num_continuous, dim=32, dim_out=1, depth=6, heads=8, attn_dropout=0.1, ff_dropout=0.1, mlp_hidden_mults=(4, 2), mlp_act=nn.ReLU()):
        super().__init__()
        try:
            self.categories = categories
            self.num_continuous = num_continuous
            self.dim = dim
            self.depth = depth
            self.heads = heads
            self.attn_dropout = attn_dropout
            self.ff_dropout = ff_dropout
            self.mlp_hidden_mults = mlp_hidden_mults
            self.mlp_act = mlp_act
            self.column_embed = ColumnEmbedding(categories, dim)
            self.transformer = nn.Sequential(*[
                TransformerBlock(dim, heads, attn_dropout, ff_dropout)
                for _ in range(depth)
            ])
            mlp_in = dim * len(categories) + num_continuous
            mlp_layers = []
            prev = mlp_in
            for mult in mlp_hidden_mults:
                mlp_layers.append(nn.Linear(prev, mult * mlp_in))
                mlp_layers.append(mlp_act)
                prev = mult * mlp_in
            mlp_layers.append(nn.Linear(prev, dim_out))
            self.mlp = nn.Sequential(*mlp_layers)
            logger.info(f"Initialized TabTransformer: {locals()}")
        except Exception as e:
            logger.error(f"Error initializing TabTransformer: {e}")
            raise

    def forward(self, x_categ, x_cont):
        try:
            # x_categ: (batch, num_categ), x_cont: (batch, num_cont)
            logger.debug(f"TabTransformer input shapes: x_categ={x_categ.shape}, x_cont={x_cont.shape}")
            cat_emb = self.column_embed(x_categ)  # (batch, num_categ, dim)
            trans_out = self.transformer(cat_emb)  # (batch, num_categ, dim)
            flat = trans_out.flatten(1)  # (batch, num_categ * dim)
            x = torch.cat([flat, x_cont], dim=1)  # (batch, num_categ * dim + num_cont)
            out = self.mlp(x)
            logger.debug(f"TabTransformer output shape: {out.shape}")
            return out
        except Exception as e:
            logger.error(f"Error in TabTransformer forward: {e}")
            raise 