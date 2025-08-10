import torch
import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    """
    Embedding layer for tokens.
    """
    def __init__(self, vocab_size, embed_size):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class SegmentEmbedding(nn.Embedding):
    """
    Embedding layer for segment (sentence) ids.
    """
    def __init__(self, n_segments, embed_size):
        super().__init__(n_segments, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Embedding):
    """
    Embedding layer for positional encodings, optionally with a padding index.
    """
    def __init__(self, max_len, embed_size, padding_idx=None):
        if padding_idx is not None:
            super().__init__(max_len, embed_size, padding_idx=padding_idx)
        else:
            super().__init__(max_len, embed_size)

class BERTEmbedding(nn.Module):
    """
    Combines token, segment, and positional embeddings for BERT-style models.
    """
    def __init__(self, vocab_size, n_segments, max_len, embed_size, dropout):
        super().__init__()
        self.tok_embed = TokenEmbedding(vocab_size, embed_size)
        self.seg_embed = SegmentEmbedding(n_segments, embed_size)
        self.pos_embed = PositionalEmbedding(max_len, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, seg_ids):
        """
        Args:
            x: Tensor of token ids (batch, seq_len)
            segment: Tensor of segment ids (batch, seq_len)
        Returns:
            Embedded tensor (batch, seq_len, embed_size)
        """
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = self.tok_embed(x) + self.seg_embed(seg_ids) + self.pos_embed(pos)
        return self.dropout(x)