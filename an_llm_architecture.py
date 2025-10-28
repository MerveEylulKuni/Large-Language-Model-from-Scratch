GPT_CONFIG_124M = {
    "vocab_size": 50257, # size of the vocabulary
    "context_length": 1024, # context length
    "emb_dim": 768, # embedding dimension
    "n_layers": 12, # number of transformer blocks
    "n_heads": 12, # number of attention heads
    "drop_rate": 0.1, # dropout rate
    "qkv_bias": False # query, key, value bias
}

# A placeholder gpt model architecture class
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(  # uses a placeholder for TransformerBlock                      
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
           
        self.final_norm = DummyLayerNorm(cfg["emb_dim"]) # uses a placeholder for LayerNorm
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False) # linear output layer

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # token embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # position embeddings
        x = tok_embeds + pos_embeds  # combine token and position embeddings
        x = self.drop_emb(x)  # apply dropout
        x = self.trf_blocks(x)  # pass through transformer blocks
        x = self.final_norm(x)  # final layer normalization
        logits = self.out_head(x)  # output logits
        
        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
    def forward(self, x): # this block does nothing, just returns its input
        return x     
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        
    def forward(self, x): 
        return x  

# Prepare an input data and test the model    
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"        
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0) 
print("Batch of the example input data:\n", batch) # the first row corresponds to txt1, the second to txt2

# Initialize the dummy gpt instance and feed it the tokenized batch
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape) 
print("Output (logits)\n", logits) # output tensor has 2 rows, one for each text in the batch.
# each text sample consists of 4 tokens. Each token is a 50257 dimensional vector which matches the size of the tokenizer vocab.

# Normalizing activations with layer normalization
# Starting with the real layer normalization class we will replace the dummy one in the model above.
torch.manual_seed(123)
batch_example = torch.randn(2, 5) # a batch of 2 samples, each with 5 features (dimensions)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print("Output before layer normalization:\n", out) # first row  corresponds to the first sample, the second to the second sample

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# Applying layer normalization to layer outputs
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Output after layer normalization:\n", out_norm)
# mean and variance after normalization
print("Mean after layer normalization:\n", mean)
print("Variance after layer normalization:\n", var)

# Turn off scientific notation for readability
torch.set_printoptions(sci_mode=False)
print("Mean after layer normalization (readable):\n", mean)
print("Variance after layer normalization (readable):\n", var)

# A layer normalization class
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
    
# Try the layer normalization class in practice
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
print("\nMean after using module layer normalization:\n", mean)
print("Variance after using module layer normalization:\n", var)

# Implement a GELU activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))    
    
# Compare ReLU and GELU activations
import matplotlib.pyplot as plt
gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100) # creates 100 data points between -3 and 3   
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label} (x)")
    plt.grid(True)
plt.tight_layout()
plt.show()    

# Use gelu to implement the small neural network module, feedforward
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
    
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.randn(2, 3, 768) # 2 batches, each with 3 tokens, each token represented by a 768-dimensional vector (emb_dim)
out = ffn(x)
print("\nOutput shape of the feedforward module:", out.shape)    

# A neural network to illustrate the shortcut connections
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__() 
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([ # implement 5 layers
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), nn.GELU()),
        ])
        
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x) # output of the current layer
            if self.use_shortcut and x.shape == layer_output.shape: # check if shortcut can be applied
                x = x + layer_output # add the input to the output (shortcut connection)
            else:
                x = layer_output # no shortcut, just take the output
        return x
    
# Initialize a neural network without shortcut connections
layer_sizes = [3, 3, 3, 3, 3, 1] # sizes of each layer
sample_input = torch.tensor([1., 0., -1.])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

print() # new line for better readability

# Implement a function that computes the gradients during the model's backward pass
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([0.]) # target value for loss computation
    
    loss = nn.MSELoss() 
    loss = loss(output, target) # compute the loss based on how close the output and target are
    
    loss.backward() # backward pass to compute the gradients
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
 
print_gradients(model_without_shortcut, sample_input)   
print() # new line for better readability

# Initialize a model with skip connections
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)      

# CONNECTING ATTENTION MECHANISM AND LINEAR LAYERS TO BUILD A TRANSFORMER BLOCK
# The transformer block component of the GPT architecture
from causal_and_multihead_attention import MultiHeadAttention

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

# Instantiate the transformer block with a sample input
torch.manual_seed(123)
x = torch.randn(2, 4, 768) # creates sammple input of shape [batch, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("\nInput shape:", x.shape)
print("Output shape:", output.shape)

# Next:  FINAL GPT MODEL ARCHITECTURE (coding the gpt model)