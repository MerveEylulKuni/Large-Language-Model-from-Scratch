# CALCULATING THE TRAINING AND VALIDATION SET LOSSES
import torch
import tiktoken

from gpt_model import GPTModel, generate_text_simple
from pretraining_on_unlabeled_data import text_to_token_ids, token_ids_to_text

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, #we shorten the context length from 1024 to 256 tokens
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1, #it's possible and common to set the dropout to 0
    "qkv_bias": False
}    

#implement a utility function to compute the cross entropy loss of a given batch returned via training and validation loader
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device) #the transfer to a given device allows us to transfer the data to a gpu
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    
    return loss #computes the loss for a single batch

#using the train data and validation data subsets, create the respective data loader
from torch.utils.data import Dataset, DataLoader
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

#use the utility function above to implement calc_loss_loader which computes the loss over all the batches sampled by a given data loader
# Function to compute the training and validation loss
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader) #iterates over all batches if no fixed num_batches specified        
    else:   
        num_batches = min(num_batches, len(data_loader)) #reduces the number of batches to match the total number of batches in the data loader  
    #if  num_batches exceeds the number of batches in the data loader
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item() #sums loss for each batch
        else:
            break
    
    return total_loss / num_batches #averages the loss over all batches  

def create_train_val_loaders_from_text(text, config,
                                       batch_size_train=2, batch_size_val=2,
                                       max_length=None, stride=None,
                                       drop_last_train=True, drop_last_val=False,
                                       num_workers=0):
    
    max_length = max_length or config["context_length"]
    stride = stride or max_length
    
    train_ratio = 0.90 
    split_idx = int(train_ratio * len(text))
    train_data = text[:split_idx]
    val_data = text[split_idx:]
    
    train_loader = create_dataloader_v1(
            train_data,
            batch_size=2,
            max_length=GPT_CONFIG_124M["context_length"],
            stride=GPT_CONFIG_124M["context_length"],
            drop_last=True,
            shuffle=True,
            num_workers=0
        ) 
    val_loader = create_dataloader_v1(
            val_data,
            batch_size=2,
            max_length=GPT_CONFIG_124M["context_length"],
            stride=GPT_CONFIG_124M["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0
        )
    return train_loader, val_loader

def create_train_val_loaders_from_file(file_path, config, **kwargs):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return create_train_val_loaders_from_text(text, config, **kwargs)

def main():
    torch.manual_seed(456)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
        
    # Instantiate the BPE tokenizer from tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
        
    total_characters = len(text_data)    
    total_tokens = len(tokenizer.encode(text_data))
    print("\nCharacters:", total_characters)
    print("Tokens:", total_tokens)

    train_loader, val_loader = create_train_val_loaders_from_file(
        file_path,
        GPT_CONFIG_124M,
        batch_size_train=2,
        batch_size_val=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last_train=True,
        drop_last_val=False,
        num_workers=0
    )

    torch.manual_seed(456)   

    #as an optional check, iterate through the data loaders to make sure that they were created correctly
    print("\nTrain loader:")
    for x, y in train_loader:
        print (x.shape, y.shape)
        
    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad(): #disables gradient tracking for efficiency because we are not training yet
        train_loss = calc_loss_loader(train_loader, model, device) #via the device setting we ensure the data is loaded onto the same device as the llm model
        val_loss = calc_loss_loader(val_loader, model, device)
    print("\nTraining loss:", train_loss)
    print("Validation loss:", val_loss)     

if __name__ == "__main__":
    main()
# Next: training an llm