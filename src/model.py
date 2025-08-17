import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings import BERTEmbedding
from src.attention import MultiHeadAttention
from src.utils import SublayerConnection, PositionWiseFeedForward

class EncoderLayer(nn.Module):
    """
    Transformer encoder layer.
    """
    def __init__(self, attn_heads, embed_size, ff_size, dropout=0.1):
        """
        Args:
            attn_heads (int): Number of attention heads.
            embed_size (int): Embedding dimension.
            ff_size (int): Feedforward layer dimension.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.attention = MultiHeadAttention(attn_heads, embed_size, dropout)
        self.feed_forward = PositionWiseFeedForward(embed_size, ff_size, dropout)
        self.sublayers = nn.ModuleList([SublayerConnection(embed_size, dropout) for _ in range(2)])

    def forward(self, x, mask):
        """
        Args:
            x (Tensor): Input tensor.
            mask (Tensor): Attention mask.
        Returns:
            Tensor: Output tensor.
        """
        x = self.sublayers[0](x, lambda x: self.attention(x, x, x, mask))
        x = self.sublayers[1](x, self.feed_forward)
        return x

class BERT(nn.Module):
    """
    BERT encoder stack for masked language modeling and NSP.
    """
    def __init__(self, vocab_size, n_segments, max_len, attn_heads, embed_size, ff_size, n_layers, dropout):
        """
        Args:
            vocab_size (int): Vocabulary size.
            n_segments (int): Number of segment types.
            max_len (int): Maximum sequence length.
            attn_heads (int): Number of attention heads.
            embed_size (int): Embedding dimension.
            ff_size (int): Feedforward layer dimension.
            n_layers (int): Number of encoder layers.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.embed_size = embed_size
        self.embedding = BERTEmbedding(vocab_size, n_segments, max_len, embed_size, dropout)
        self.encoder = nn.ModuleList([
            EncoderLayer(attn_heads, embed_size, ff_size, dropout) for _ in range(n_layers)    
        ])
        # Apply BERT initialization to all submodules
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights as in the original BERT paper:
        - All weights: normal(0, 0.02)
        - All biases: zeros
        - LayerNorm gamma: ones, beta: zeros (PyTorch default)
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, segment_ids, attention_mask=None):
        """
        Args:
            input_ids (Tensor): Input token ids (batch_size, seq_len)
            segment_ids (Tensor): Segment ids (batch_size, seq_len)
            attention_mask (Tensor, optional): Attention mask (batch_size, seq_len). Defaults to None.
        Returns:
            Tensor: Encoded hidden states (batch_size, seq_len, embed_size)
        """
        if attention_mask is None:
            mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)
        else:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        x = self.embedding(input_ids, segment_ids)
        for layer in self.encoder:   
            x = layer(x, mask)
        return x

class MaskedLanguageModel(nn.Module):
    """
    Head for masked language modeling (MLM) in BERT.
    """
    def __init__(self, embedding_layer, vocab_size):
        """
        Args:
            embedding_layer (nn.Embedding): Token embedding layer to tie weights with.
            vocab_size (int): Vocabulary size.
        """
        super().__init__()
        self.embedding_layer = embedding_layer
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x (Tensor): Hidden states (batch_size, seq_len, embed_size)
        Returns:
            Tensor: Log-probabilities (batch_size, seq_len, vocab_size)
        """
        logits = F.linear(x, self.embedding_layer.weight, self.bias)
        return self.softmax(logits)

class NextSentencePrediction(nn.Module):
    """
    Head for next sentence prediction (NSP) in BERT.
    """
    def __init__(self, embed_size):
        """
        Args:
            embed_size (int): Embedding dimension.
        """
        super().__init__()
        self.linear = nn.Linear(embed_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x (Tensor): Hidden states (batch_size, seq_len, embed_size)
        Returns:
            Tensor: Log-probabilities (batch_size, 2)
        """
        return self.softmax(self.linear(x[:, 0]))   # Use [CLS] token

class BERTLM(nn.Module):
    """
    BERT language model with MLM and NSP heads.
    """
    def __init__(self, bert, vocab_size):
        """
        Args:
            bert (BERT): BERT model instance.
            vocab_size (int): Vocabulary size.
        """
        super().__init__()
        self.bert = bert
        # Pass the token embedding layer to MLM head for weight tying
        self.mlm = MaskedLanguageModel(bert.embedding.tok_embed, vocab_size)
        self.nsp = NextSentencePrediction(bert.embed_size)

    def forward(self, input_ids, seg_ids, attention_mask=None):
        """
        Args:
            x (Tensor): Input token ids (batch_size, seq_len)
            seg_ids (Tensor): Segment ids (batch_size, seq_len)
        Returns:
            Tuple[Tensor, Tensor]: (MLM log-probs, NSP log-probs)
        """
        hidden = self.bert(input_ids, seg_ids, attention_mask)
        return self.nsp(hidden), self.mlm(hidden)
    
class BERT4GLUE(nn.Module):
    """
    BERT model for GLUE tasks.
    """
    def __init__(self, bert, task_name, dropout=0.1):
        super().__init__()
        assert task_name in ["ax", "mnli", "qqp", "qnli", "sst2", "cola", "stsb", "mrpc", "rte", "wnli"], "task_name not a GLUE task"
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        embed_size = bert.embed_size
        self.head = nn.Linear(
            embed_size,
            3 if task_name in ("mnli", "ax") else
            1 if task_name=="stsb" else
            2
        )

    def forward(self, input_ids, seg_ids, attention_mask=None):
        hidden = self.bert(input_ids, seg_ids, attention_mask)
        cls = hidden[:, 0]
        logits = self.head(self.dropout(cls))
        return logits