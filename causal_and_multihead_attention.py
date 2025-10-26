import torch
import torch.nn as nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], #Your
     [0.55, 0.87, 0.66], #journey
     [0.57, 0.85, 0.64], #starts
     [0.22, 0.58, 0.33], #with
     [0.77, 0.25, 0.10], #one
     [0.05, 0.80, 0.55]] #step
)

# from IMPLEMENTING SELF-ATTENTION WITH TRAINABLE WEIGHTS dim_in and dim_out

x_2 = inputs[1] # the second input element
dim_in = inputs.shape[1] # the input embedding size, d=3
dim_out = 2 # the output embedding size, d=2

class SelfAttention_v2(nn.Module):
    def __init__(self, dim_in, dim_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)       
        queries = self.W_query(x)  
        values = self.W_value(x)  

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vecs = attn_weights @ values
        return context_vecs

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(dim_in, dim_out)
print(sa_v2(inputs)) 


# HIDING FUTURE WORDS WITH CAUSAL ATTENTION

# Applying a causal attention mask
# First, we compute the attention weights as before:
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax( attn_scores / keys.shape[-1]**0.5, dim=-1 )
print("Attention weights:\n", attn_weights)

# Use the pytorch's tril frunction to create a mask where the values above the diagonal are 0
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones((context_length, context_length)))
print("Attention masked:\n", mask_simple)

masked_simple = attn_weights * mask_simple
print("Masked attention weights:\n", masked_simple)

# Re-normalize the masked attention weights so that they sum to 1
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple = masked_simple / row_sums
print("Masked and re-normalized attention weights:\n", masked_simple)

# Mask the attention scores with negative infinity values before applying softmax
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("Attention scores masked with -inf:\n", masked)

# Apply softmax to these masked scores
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print("Masked attention weights:\n", attn_weights)


# MASKING ADDITIONAL ATTENTION WEIGHTS WITH DROPOUT

# Apply pytorch's nn.Dropout module first to a 6 x 6 tensor consisting of all ones for simplicity
torch.manual_seed(456)
dropout = torch.nn.Dropout(0.5) #50% dropout rate
example = torch.ones(6, 6) #created a matrix of 1s
print("Example dropout tensor:\n", dropout(example))

# Apply dropout to the attention weight matrix itself
torch.manual_seed(123)
print("Attention weight matrix with dropout:\n", dropout(attn_weights)) 

# Implementing a compact causal attention class
batch = torch.stack((inputs, inputs), dim=0)
print("Batch shape:", batch.shape) # (2, 6, 3)

class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.dim_out = dim_out
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, dim_in = x.shape
        keys = self.W_key(x)       
        queries = self.W_query(x)  
        values = self.W_value(x)  

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool() [:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vecs = attn_weights @ values
        return context_vecs
    
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(dim_in, dim_out, context_length, 0.0)
context_vecs = ca(batch)
print("Context vectors shape:", context_vecs.shape) # (2, 6, 2)    


# MULTI-HEAD ATTENTION

# Stacking multiple single-head attention modules
# A wrapper class to implement multi-head attention
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(dim_in, dim_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
context_length = batch.shape[1] #this is the number of tokens, 6 in this case
dim_in, dim_out = 3, 2
mha = MultiHeadAttentionWrapper(dim_in, dim_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print("Context vectors shape:", context_vecs.shape) # (2, 6, 4)
print("Context vectors:\n", context_vecs)

# Implementing multi-head attention with weight splits
# An efficent multi-head attention class
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
    
torch.manual_seed(123)
batch_size, context_length, dim_in = batch.shape
dim_out = 2
mha = MultiHeadAttention(dim_in, dim_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print("Context vector with multi-head attention:\n", context_vecs)
print("context_vecs.shape:", context_vecs.shape)    

# Next: an_llm_architecture.py