import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. Механизм внимания
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, seq_length, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim должен делиться на n_heads"
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            'mask', 
            torch.tril(torch.ones(seq_length, seq_length))
            .view(1, 1, seq_length, seq_length)
        )
    
    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        y = self.proj(y)
        
        return y


# 2. Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

# 3. Transformer Block (Attention + FFN + Norm)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, seq_length, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_dim, n_heads, seq_length, dropout)
        self.feed_forward = FeedForward(embed_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Pre-norm архитектура (современный стандарт)
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

# 4. Полная LLM модель
class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, 
                 seq_length, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_length, embed_dim)
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, seq_length, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.seq_length, f"Длина последовательности {T} > {self.seq_length}"
        
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.dropout(token_emb + pos_emb)
        
        x = self.blocks(x)
        x = self.norm(x)
        
        logits = self.head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Генерация нового текста"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_length:]
            
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx