import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

class SelfAttentionHead(nn.Module):
    def __init__(self, config):
        super(SelfAttentionHead, self).__init__()
        self.config = config

        self.query = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.key = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.attn_drop = nn.Dropout(config.attn_drop_value)
        self.register_buffer('tril', torch.tril(torch.ones(config.context_len, config.context_len)))

    def forward(self, x):
        # x.shape: (Batch, Context Length, Embedding Dimension)
        B, C, N = x.shape
        q = self.query(x) # (B, C, head_size)
        k = self.key(x) # (B, C, head_size)
        v = self.value(x) # (B, C, head_size)

        # Compute Attention scores
        # (B, C, head_size) bmm (B, head_size, C) -> (B, C, C)
        attn_weight = torch.div(torch.bmm(q, k.permute(0, 2, 1)), self.config.head_size)
        attn_weight = attn_weight.masked_fill(self.tril[:C, :C] == 0, float('-inf'))
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = self.attn_drop(attn_weight)

        # Do weighted aggregation of values
        output = torch.bmm(attn_weight, v)
        return output

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadedAttention, self).__init__()
        self.head_size = config.head_size
        self.num_heads = config.num_heads
        self.embed_dim = config.n_embed
    
        self.heads = nn.ModuleList(
            [SelfAttentionHead(config) for _ in range(self.num_heads)]
        )
        self.proj = nn.Linear(config.num_heads * config.head_size, config.n_embed)
        self.drop = nn.Dropout(config.multihead_drop_value)
    
    def forward(self, x):
        multihead_output = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.drop(self.proj(multihead_output))
    
class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed * 4),
            nn.GELU(),
            nn.Linear(config.n_embed * 4, config.n_embed),
            nn.Dropout(config.ffn_drop_value)
        )
    def forward(self, x):
        return self.ffn(x)
    
class GPTBlock(nn.Module):
    def __init__(self, config):
        super(GPTBlock, self).__init__()
        self.multiheaded_attn = MultiHeadedAttention(config)
        self.ffn = FFN(config)
        self.layernorm1 = nn.LayerNorm(config.n_embed)
        self.layernorm2 = nn.LayerNorm(config.n_embed)
    
    def forward(self, x):
        x = x + self.layernorm1(self.multiheaded_attn(x))
        x = x + self.layernorm2(self.ffn(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config
        # Init layers and stuff
        self.tok_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embedding = nn.Embedding(config.context_len, config.n_embed)
        self.blocks = nn.Sequential(*[GPTBlock(config) for _ in range(config.num_blocks)])
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
    
    def forward(self, x, targets=None):
        # Input is just tokenized text of 'B' batches, each 'C' context length long
        B, C = x.shape
        
        # First we apply the token embedding -> tok_emb (B, C, V)
        tok_emb = self.tok_embedding(x)
        # Then we get the positional embeddings with length equal to context len
        pos_emb = self.pos_embedding(torch.arange(C, device=self.config.device))
        # Then we add them
        x = tok_emb + pos_emb
        # Then we pass the input through all the GPT blocks
        x = self.blocks(x)
        # And finally pass it through the final layer to get the logits
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, C, V = logits.shape
            logits = logits.view(B*C, V)
            targets = targets.view(B*C)
            loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss
    
def generate(model, prompt, max_tokens, temperature=0.7):
    """
    Generates text based on given prompt
    """
    for _ in range(max_tokens):
        prompt = prompt[:, :Config.context_len]
        logits, _ = model(prompt)
        logits = logits[:, -1, :]
        logits = logits / temperature
        logit_probs = nn.functional.softmax(logits, dim=-1)
        next_prompt = torch.multinomial(logit_probs, num_samples=1)
        prompt = torch.cat((prompt, next_prompt), dim=1)
    return prompt