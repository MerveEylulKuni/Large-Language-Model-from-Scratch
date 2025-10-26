# A SIMPLE SELF-ATTENTION MECHANISM WITHOUT TRAINABLE WEIGHTS

import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], #Your
     [0.55, 0.87, 0.66], #journey
     [0.57, 0.85, 0.64], #starts
     [0.22, 0.58, 0.33], #with
     [0.77, 0.25, 0.10], #one
     [0.05, 0.80, 0.55]] #step
)
     
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])     
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)     
print("Attention scores:", attn_scores_2)    

# Normalize the attention scores
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum of attention weights:", attn_weights_2_tmp.sum())

# Softmax version of the attention scores
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim = 0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Softmax attention weights:", attn_weights_2_naive)
print("Sum of softmax attention weights:", attn_weights_2_naive.sum())

# Use the pytorch implementation of softmax
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Softmax attention weights (pytorch):", attn_weights_2)
print("Sum of softmax attention weights (pytorch):", attn_weights_2.sum())

# Calculate the context vector as a weighted sum of the input vectors
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("Context vector:", context_vec_2)

# Computing attention weights for all input tokens
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)    
print("Attention scores:\n", attn_scores)

# Since for loops are slow better to use matrix multiplication to achieve the same result
attn_scores = inputs @ inputs.T
print("Attention scores (with matmul):\n", attn_scores)     

# Normalize each row of the attention scores matrix
attn_weights = torch.softmax(attn_scores, dim=-1)
print("Attention weights:\n", attn_weights)   

# Verify that the rows sum to 1
row_4_sum = sum([0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295])
print("Row 4 sum:", row_4_sum)
print("Sum of each row of attention weights:\n", attn_weights.sum(dim=-1))

# Use attention weights to compute context vectors
all_context_vecs = attn_weights @ inputs
print("All context vectors:\n", all_context_vecs) 

# Double check the correctness by comparing the 2nd row with the context vector
print("Previous 2nd context vector:\n", context_vec_2)


# IMPLEMENTING SELF-ATTENTION WITH TRAINABLE WEIGHTS

x_2 = inputs[1] # the second input element
dim_in = inputs.shape[1] # the input embedding size, d=3
dim_out = 2 # the output embedding size, d=2

# Initialize the 3 weight matrices: query, key, value
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=False)

# Compute the query, key, and value vectors
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print("Query 2 vector:", query_2)

#  Obtain all keys and values via matmul
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# Compute the attention scores
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print("Attention score:", attn_score_22)

# Generalize this computation to all attention scores
attn_scores_2 = query_2 @ keys.T
print("Attention scores:", attn_scores_2)

# Compute the attention weights
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("Attention weights:", attn_weights_2)

# Compute the context vector as a weighted sum of the value vectors
context_vec_2 = attn_weights_2 @ values
print("Context vector:", context_vec_2)


# IMPLEMENTING A COMPACT SELF-ATTENTION PYTHON CLASS

# A compact self-attention class
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, dim_in, dim_out): # init method initializes trainable weight matrices (W_query, W_key, W_value)
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(dim_in, dim_out))
        self.W_key = nn.Parameter(torch.rand(dim_in, dim_out))
        self.W_value = nn.Parameter(torch.rand(dim_in, dim_out))

    def forward(self, x):
        keys = x @ self.W_key       
        queries = x @ self.W_query  
        values = x @ self.W_value  

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vecs = attn_weights @ values
        return context_vecs
    
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(dim_in, dim_out)   
print(sa_v1(inputs)) 

# A self-attention class using pytorch's built-in nn.Linear layer
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

# Next: causal_and_multihead_attention.py 