# USING GPT TO GENERATE TEXT
import torch
import torch.nn as nn

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
    
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, #we shorten the context length from 1024 to 256 tokens
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1, #it's possible and common to set the dropout to 0
    "qkv_bias": False
}    

# Implement the text generation process
# Utility functiond for text to token id conversion
import tiktoken

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

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #unsqueeze adds the batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) #removes the batch dimension
    return tokenizer.decode(flat.tolist())

def main():
    
    torch.manual_seed(456)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    print()


    # CALCULATING THE TEXT GENERATON LOSS

    inputs = torch.tensor([[16833, 3626, 6100],   #["every effort moves",
                        [40   , 1107,  588]])  # "I really like"]
    targets = torch.tensor([[3626, 6100,   345],  # ["effort moves you", 
                            [1107,  588, 11311]]) #  "really like chocolate"]

    with torch.no_grad(): #disables gradient tracking since we are not training yet
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1) #probability of each token in vocabulary
    print(probas.shape) #the first number, 2 corresponds to the two examples in the inputs, also known as batch size
    #the second number, 3 to the number of tokens in each row (input) 
    #the last number to the embedding dimensionality, which is determined by the vocabulary size

    #applying the argmax function to the probability scores to obtain the corresponding token ids
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("\nToken IDs:\n", token_ids)

    #convert the token ids back into text
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1:"
        f"{token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    #for each of the two input texts, print the initial softmax probability scores corresponding to the target tokens
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("\nText 1:", target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)

    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)

    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)

    #recall the shape of the logits and target tensors before applying cross entropy function
    print("\nLogits shape:", logits.shape) #batch size, number of tokens, vocabulary size
    print("Targets shape:", targets.shape) #batch size, number of tokens

    #for the cross entropy function in pytorch, flatten these tensors by combining them over the batch dimension
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("\nFlattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print("\nLoss:", loss)

    perplexity = torch.exp(loss)
    print("Perplexity:", perplexity)

if __name__ == "__main__":
    main()
# Next: training and validation set losses