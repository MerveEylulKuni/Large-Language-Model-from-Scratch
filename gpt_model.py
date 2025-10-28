# FINAL GPT MODEL ARCHITECTURE (coding the gpt model)
import torch
import torch.nn as nn

# Define model configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257, # size of the vocabulary
    "context_length": 1024, # context length
    "emb_dim": 768, # embedding dimension
    "n_layers": 12, # number of transformer blocks
    "n_heads": 12, # number of attention heads
    "drop_rate": 0.1, # dropout rate
    "qkv_bias": False # query, key, value bias
} 

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # small constant added to the variance to avoid division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim)) # trainable parameter
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # trainable parameter
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift  

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))    
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # first linear layer
            GELU(), # GELU activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), # second linear layer
        )
        
    def forward(self, x):
        return self.layers(x)    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (dim_out % num_heads == 0), \
            "dim_out must be divisible by num_heads"
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads #reduces the projection dim to match the desired output dim

        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.out_proj = nn.Linear(dim_out, dim_out) #uses a linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, dim_in = x.shape

        queries = self.W_query(x)    # |\
        keys = self.W_key(x)         # | --> tensor shape: (b, num_tokens, dim_out)  
        values = self.W_value(x)     # |/
        
        #implicitly split the matrix by adding a num_heads dimension.
        #then we unroll the last dim: (b, num_tokens, dim_out) -> (b, num_tokens, num_heads, head_dim) 
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2) #transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3) #computes dot product attention for each head
        mask_bool = self.mask.bool() [:num_tokens, :num_tokens] #masks truncated to the number of tokens
        
        attn_scores.masked_fill_(mask_bool, -torch.inf) #uses the mask to fill attention scores
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vecs = (attn_weights @ values).transpose(1, 2) #tensor shape: (b, num_tokens, num_heads, head_dim)
        context_vecs = context_vecs.contiguous().view(b, num_tokens, self.dim_out) #combines heads, where self.dim_out = self.num_heads * self.head_dim
        context_vecs = self.out_proj(context_vecs) #adds an optional linear projection
        
        return context_vecs

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg["emb_dim"])
        self.att = MultiHeadAttention(
            dim_in=cfg["emb_dim"],
            dim_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        
    def forward(self, x):
        shortcut = x # shortcut connection for attention block
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut #add the original input back
        
        shortcut = x # shortcut connection for feedforward block
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x    

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential( *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])  # stabilizes the learning process
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # linear output layer

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # token embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) #the device setting allows us to train the model on GPU or CPU, depending on which device the input data sits on.
        x = tok_embeds + pos_embeds  # combine token and position embeddings
        x = self.drop_emb(x)  # apply dropout
        x = self.trf_blocks(x)  # pass through transformer blocks
        x = self.final_norm(x)  # final layer normalization
        logits = self.out_head(x)  # output logits

        return logits   
    
# GENERATING TEXT
# A function for the gpt model to generate text
def generate_text_simple(model, idx, max_new_tokens, context_size): #idx is a (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] #crops context size if it exceeds the supported context size
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :] #focuses only on the last time step, so (batch, n_token, vocab_size) becomes (batch, vocab_size)   
        probas = torch.softmax(logits, dim=-1) #probas has shape (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) #idx_next has shape (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1) #appends sampled index to the running sequence, where idx has shape (batch, n_tokens+1)
    return idx     

def main():    
# Initialize the model
    torch.manual_seed(123) 
    model = GPTModel(GPT_CONFIG_124M)

    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"        
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0) 

    out = model(batch)
    print("\nInput batch:\n", batch)
    print("Output shape:", out.shape)
    print("Logits:\n", out)

    # Before moving onto the function that converts logits into text,
    # collect the total number of parameters 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params:,}")

    print("\nToken embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)
    print()

    # Remove the output layer parameter count from the total gpt2 model count according to the weight tying
    total_params_gpt2 = (
        total_params - sum(p.numel() for p in model.out_head.parameters())
    )
    print(f"Number of trainable parameters "
        f"considering weight tying: {total_params_gpt2:,}")

    # Compute the memory requirements of the 163 million parameters in our gptmodel object
    total_size_bytes = total_params * 4 # 4 bytes per parameter (float32), calculate total size in bytes
    total_size_mb = total_size_bytes / (1024 * 1024) # convert bytes to megabytes
    print(f"\nTotal size of the model: {total_size_mb:.2f} MB")

    # Try the function, first encode the input context into token ids
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("\nEncoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #adds batch dimension
    print("Encoded tensor shape:", encoded_tensor.shape)    

    # Put the model into eval() mode
    model.eval() #disables dropout since we are not training the model
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("\nOutput:", out)
    print("Output length:", len(out[0]))

    # Convert the ids back into text
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)
    
if __name__ == "__main__":
    main()
    
# Next: pretraining_on_unlabeled_data.py