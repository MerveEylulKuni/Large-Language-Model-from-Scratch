# BYTE PAIR ENCODING (BPE)
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

# Instantiate the BPE tokenizer from tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# Usage of this tokenizer is similar to the SimpleTokenizerV2 class above
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Token IDs:", integers)

strings = tokenizer.decode(integers)
print("Decoded text:", strings) 


# DATA SAMPLING WITH A SLIDING WINDOW

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
enc_text = tokenizer.encode(raw_text)
print(f"Length of text in characters: {len(enc_text)}")   

# Remove the first 50 tokens from the dataset for demonstration purposes
enc_sample = enc_text[50:]

# Define the sliding window sampling function.
# One of the easiest ways to create the input-target pairs for the next word prediction task, 
# is to create two variables, x and y, where x contains the input tokens and y contains the targets, which are the input tokens shifted by one. 
context_size = 4 # the context size determines how many tokens are included in the input.
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"when input is {context} the desired output is {desired}")
    
# Lets repeat the previous code but convert the token ids into text.
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"when input is -->( {tokenizer.decode(context)} ), the desired output is -->( {tokenizer.decode([desired])} ) ")
    
# A dataset for batched inputs and targets
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt) # tokenizes the entire text
        
        for i in range (0, len(token_ids) - max_length, stride): # uses a sliding window to chunk the book into overlapping sequences of max_length
            input_chunk = token_ids[i : i+max_length]
            target_chunk = token_ids[i+1 : i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self): # returns the total number of rows in the dataset
        return len(self.input_ids)
    
    def __getitem__(self, idx): # returns a single row from the dataset
        return self.input_ids[idx], self.target_ids[idx]
    
# A data loader to generate batches with input-with pairs
def create_dataloader_v1(txt, batch_size=4,max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
        tokenizer = tiktoken.get_encoding("gpt2") # initializes the tokenizer
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) # creates dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last, # drop the last batch if it's shorter than the specified batch_size to prevent loss spikes during training.
            num_workers=num_workers # the number of cpu processes to use for preprocessing.
        )
        
        return dataloader
    
# Test the dataloader to develop an intuition of how the GPTDatasetV1 class and the create_dataloader_v1 function work together
#with open("the-verdict.txt", "r", encoding="utf-8") as f:
#        raw_text = f.read()
        
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    
data_iter = iter(dataloader) # converts dataloader into a python iterator to fetch the next entry via python's built-in next() function.
first_batch = next(data_iter)
print(first_batch)    

# To understand the meaning of stride=1, let's fetch another batch
second_batch = next(data_iter)
print(second_batch)

# Let's look at how we can use the data loader to sample with a batch size > 1
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


# CREATING TOKEN EMBEDDINGS

# Suppose we have the following input tokens
input_ids = torch.tensor([2, 3, 5, 1])

# For the sake of simplicity, let's assume we have a small vocabulary size of only 6 words and create embeddings of size 3
vocab_size = 6
output_dim = 3

# We can instantiate an embedding layer in PyTorch
torch.manual_seed(123) # for reproducibility
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# Let's apply it to a token id to get an embedding vector
print(embedding_layer(torch.tensor([3])))

# Now let's apply it to all four (the entire) input ids
print(embedding_layer(input_ids))


# ENCODING WORD POSITIONS

# Now consider more realistic embedding sizes and encode the input tokens into a 256-dimensional vector representation.
# We assume the token ids were created by the bpe tokenizer above, which has a vocab size of 50257
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)  

# Instantiate the data loader first
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# Use the embedding layer to embed these token ids into 256-dimensional vectors
token_embeddings = token_embedding_layer(inputs) 
print("\nToken Embeddings shape:\n", token_embeddings.shape)

# For the gpt model's absolute embedding approach, we just need to create another embedding layer 
# that has the same embedding dimension as the token_embedding_layer
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("\nPosition Embeddings shape:\n", pos_embeddings.shape)

# Now we can add the position embeddings directly to the token embeddings
# where pytorch will add the 4x256 dimensional pos_embeddings tensor to each 4x256 dimensional token_embeddings tensor in each of the 8 batches
input_embeddings = token_embeddings + pos_embeddings
print("\nInput Embeddings shape:\n", input_embeddings.shape)

# Ch2 (Working with text data) is done. NEXT: Coding attention mechanisms.