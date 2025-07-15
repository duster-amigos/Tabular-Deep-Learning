import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class FeatureTokenizer(nn.Module):
    """
    Tokenizes each feature (categorical or numerical) into a learnable embedding, as in FTTransformer.
    """
    def __init__(self, categories, num_continuous, dim):
        super().__init__()
        try:
            self.categories = categories
            self.num_continuous = num_continuous
            self.dim = dim
            self.cat_embeds = nn.ModuleList([
                nn.Embedding(num_cat + 1, dim)  # +1 for missing value
                for num_cat in categories
            ])
            self.cont_embeds = nn.Parameter(torch.randn(num_continuous, dim))
            logger.info(f"Initialized FeatureTokenizer: {len(categories)} categorical, {num_continuous} continuous, dim={dim}")
        except Exception as e:
            logger.error(f"Error initializing FeatureTokenizer: {e}")
            raise

    def forward(self, x_categ, x_cont):
        try:
            # x_categ: (batch, num_categ), x_cont: (batch, num_cont)
            batch_size = x_categ.size(0)
            cat_tokens = [emb(x_categ[:, i]) for i, emb in enumerate(self.cat_embeds)]  # list of (batch, dim)
            cat_tokens = torch.stack(cat_tokens, dim=1) if cat_tokens else None  # (batch, num_categ, dim)
            if self.num_continuous > 0:
                cont_tokens = x_cont.unsqueeze(-1) * self.cont_embeds.unsqueeze(0)  # (batch, num_cont, dim)
            else:
                cont_tokens = None
            tokens = []
            if cat_tokens is not None:
                tokens.append(cat_tokens)
            if cont_tokens is not None:
                tokens.append(cont_tokens)
            if tokens:
                result = torch.cat(tokens, dim=1)  # (batch, num_tokens, dim)
            else:
                result = None
            logger.debug(f"FeatureTokenizer output shape: {result.shape if result is not None else None}")
            return result
        except Exception as e:
            logger.error(f"Error in FeatureTokenizer forward: {e}")
            raise

class GLUFeedForward(nn.Module):
    def __init__(self, dim, ff_mult=4, dropout=0.1):
        super().__init__()
        try:
            self.fc1 = nn.Linear(dim, ff_mult * dim)
            self.fc2 = nn.Linear(dim, ff_mult * dim)
            self.fc3 = nn.Linear(ff_mult * dim, dim)
            self.dropout = nn.Dropout(dropout)
            logger.info(f"Initialized GLUFeedForward: dim={dim}, ff_mult={ff_mult}")
        except Exception as e:
            logger.error(f"Error initializing GLUFeedForward: {e}")
            raise

    def forward(self, x):
        try:
            x_proj = self.fc1(x)
            x_gate = torch.sigmoid(self.fc2(x))
            x = x_proj * x_gate
            x = self.fc3(x)
            x = self.dropout(x)
            logger.debug(f"GLUFeedForward output shape: {x.shape}")
            return x
        except Exception as e:
            logger.error(f"Error in GLUFeedForward forward: {e}")
            raise

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, attn_dropout, ff_dropout):
        super().__init__()
        try:
            self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(dim)
            self.ff = GLUFeedForward(dim, ff_mult=4, dropout=ff_dropout)
            self.norm2 = nn.LayerNorm(dim)
            logger.info(f"Initialized FTTransformerBlock: dim={dim}, heads={heads}")
        except Exception as e:
            logger.error(f"Error initializing FTTransformerBlock: {e}")
            raise

    def forward(self, x):
        try:
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
            logger.debug(f"FTTransformerBlock output shape: {x.shape}")
            return x
        except Exception as e:
            logger.error(f"Error in FTTransformerBlock forward: {e}")
            raise

class FTTransformer(nn.Module):
    """
    FTTransformer for numerical tabular data, following the original paper.
    Uses feature tokenization, transformer encoder stack with GLU, and a CLS token for output.
    """
    def __init__(self, categories, num_continuous, dim=32, dim_out=1, depth=6, heads=8, attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        try:
            self.categories = categories
            self.num_continuous = num_continuous
            self.dim = dim
            self.depth = depth
            self.heads = heads
            self.attn_dropout = attn_dropout
            self.ff_dropout = ff_dropout
            self.tokenizer = FeatureTokenizer(categories, num_continuous, dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.transformer = nn.Sequential(*[
                TransformerBlock(dim, heads, attn_dropout, ff_dropout)
                for _ in range(depth)
            ])
            self.head = nn.Linear(dim, dim_out)
            logger.info(f"Initialized FTTransformer: {locals()}")
        except Exception as e:
            logger.error(f"Error initializing FTTransformer: {e}")
            raise

    def forward(self, x_categ, x_cont):
        try:
            # x_categ: (batch, num_categ), x_cont: (batch, num_cont)
            logger.debug(f"FTTransformer input shapes: x_categ={x_categ.shape}, x_cont={x_cont.shape}")
            tokens = self.tokenizer(x_categ, x_cont)  # (batch, num_tokens, dim)
            batch_size = tokens.size(0)
            cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, dim)
            x = torch.cat([cls_token, tokens], dim=1)  # (batch, num_tokens+1, dim)
            x = self.transformer(x)  # (batch, num_tokens+1, dim)
            out = self.head(x[:, 0])  # (batch, dim_out)
            logger.debug(f"FTTransformer output shape: {out.shape}")
            return out
        except Exception as e:
            logger.error(f"Error in FTTransformer forward: {e}")
            raise 