# TRAINING AN LLM

from matplotlib import pyplot as plt
import torch
import tiktoken
from gpt_model import GPT_CONFIG_124M, GPTModel, generate_text_simple
from pretraining_on_unlabeled_data import text_to_token_ids, token_ids_to_text
from training_and_val_set_losses import calc_loss_batch, calc_loss_loader, create_train_val_loaders_from_file

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, #we shorten the context length from 1024 to 256 tokens
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1, #it's possible and common to set the dropout to 0
    "qkv_bias": False
}  

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    for epoch in range(num_epochs): #the main training loop
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() #resets loss gradients from the previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() #calculates loss gradients
            optimizer.step() #updates model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1
            
            if global_step % eval_freq == 0: #optional evaluation step
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Validation loss {val_loss:.3f}")
                
        generate_and_print_sample(model, tokenizer, device, start_context) #prints a sample text after each epoch
        
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() #dropout is disabled during evaluation for stable, reproducible results
    with torch.no_grad(): #disables gradient tracking since it is not required during evaluation, to reduce computational overhead
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    
    return train_loss, val_loss                

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " ")) #compact print format
    model.train()    

#create a simple plot that shows training and validation sets side by side
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny() #creates a second x axis that shares the same y axis
    ax2.plot(tokens_seen, train_losses, alpha=0) #invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()
    
#modify the text generation function to select the next token among these 3 non-zero probability scores to generate the next token
# A modified text generation function with more diversity
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for i in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        if top_k is not None: #filters logits with top_k sampling
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
        
        if temperature > 0.0: #applies temperature scaling
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
        else: #carries out greedy next token selection as before when temperature scaling is disabled
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        if idx_next == eos_id: #stops generating early if end-of-sequence token is encountered
            break
        
        idx = torch.cat((idx, idx_next), dim=1)            
        
    return idx  

#override the random weights with the loaded weights in the params dictionary
#for this define a utility function that checks whether 2 tensors or arrays have the same dimensions or shape
#and returns the right tensor as trainable pytorch parameters
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                  "Right: {right.shape}")
    
    return torch.nn.Parameter(torch.tensor(right))          
    
import numpy as np    
def load_weights_into_gpt(gpt, params):          
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])): #iterate oves each transformer block
        q_w, k_w, v_w = np.split(                           
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
        
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])
        
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])
        
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])     
    
def main():   
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    train_loader, val_loader = create_train_val_loaders_from_file(
        "the-verdict.txt",
        GPT_CONFIG_124M,
        batch_size_train=2,
        batch_size_val=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last_train=True,
        drop_last_val=False,
        num_workers=0
    )
    
    # Train a gptmodel instance for 10 epochs using adamw optimizer
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M) 
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) #the .parameters() method returns all trainable parameters of the model
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader,  optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Evert effort moves you", tokenizer=tokenizer)
     
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)   


    # DECODING STRATEGIES TO CONTROL RANDOMNESS

    model.to("cpu")
    model.eval()

    #plug gpt instance (model) into the generate_text_simple, which uses the llm to generate one token at a time
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
        model=model, 
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25, 
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))    

    # Temperature Scaling
    vocab = { 
        "closer": 0,
        "every": 1, 
        "effort": 2, 
        "forward": 3,
        "inches": 4,
        "moves": 5, 
        "pizza": 6,
        "toward": 7,
        "you": 8,
    } 
    
    inverse_vocab = {v: k for k, v in vocab.items()}

    #assume that llm is given the start context and generates the following next token logits
    next_token_logits = torch.tensor( [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

    #convert logits into probabilities
    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas).item()
    print("\nInverse vocabulary's next token id:", inverse_vocab[next_token_id])

    #to implement a probabilistic sampling process replace argmax with the ultinomial function in pytorch
    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print("Inverse vocabulary's next token id with multinomial:", inverse_vocab[next_token_id])

    #impplement a function that repeats this sampling 1000 times to see how multinomial chooses different next token sometimes
    def print_sampled_tokens(probas):
        torch.manual_seed(123)
        sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
        sampled_ids = torch.bincount(torch.tensor(sample))
        for i, freq in enumerate(sampled_ids):
            print(f"{freq} x {inverse_vocab[i]}")

    print_sampled_tokens(probas)    

    #apply temperature scaling: dividing the logits by a number greater than 0
    def softmax_with_temperature(logits, temperature):
        scaled_logits = logits / temperature
        return torch.softmax(scaled_logits, dim=0)

    #illustrate
    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize =(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')
    ax.set_ylabel('Probability')     
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig("temperature-plot.pdf")
    plt.show()

    # Top-k Sampling
    #start with the selection of the tokens with the largest logit values
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print("\nTop logits:", top_logits)
    print("Top positions:", top_pos)

    #apply pytorch's where function to set the logit values of tokens below the lowest logit value within out top-3 selection to negative infinity
    new_logits = torch.where(
        condition = next_token_logits < top_logits[-1], #identifies logits less than the min in the top 3
        input = torch.tensor(float('-inf')), #assigns -inf to these lower logits
        other = next_token_logits #retains the original logits for all other tokens
    )
    print("\nNew logits:", new_logits)

    #apply softmax to turn these logits into next-token probabilities
    topk_probas = torch.softmax(new_logits, dim=0)
    print(topk_probas)

    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Evey effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )
    print("\nOutput text:", token_ids_to_text(token_ids, tokenizer))


    # LOADING AND SAVING MODEL WEIGHTS IN PYTORCH
    from importlib.metadata import version

    pkgs = ["matplotlib", 
            "numpy", 
            "tiktoken", 
            "torch",
            "tensorflow" # For OpenAI's pretrained weights
        ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
        
    torch.save(model.state_dict(), "model.pth")

    #after saving the model weights, load the model weights into a new gptmodel instance
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
    model.eval();

    #save both the model and optimizer
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        },
        "model_and_optimizer.pth"       
    )

    #restore the model and optimizer states
    checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train();


    # LOADING PRETRAINED WEIGHTS FROM OPENAI

    print("TensorFlow version:", version("tensorflow"))
    print("tqdm version:", version("tqdm"))

    import urllib.request
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch05/"
        "01_main-chapter-code/gpt_download.py"
    )
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    
    from gpt_download import download_and_load_gpt2 #loads the gpt2 architecture settings and weight parameters
    settings, params = download_and_load_gpt2(
        model_size="124M", models_dir="gpt2"
    )

    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())

    print(params["wte"])
    print("Token embedding weight tensor dimensions:", params["wte"].shape)

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    model_name = "gpt2-small (124M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])

    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

    gpt = GPTModel(NEW_CONFIG)
    gpt.eval();

    # Loading OpenAI weights into our GPT model code
    load_weights_into_gpt(gpt, params)
    gpt.to(device); 

    torch.manual_seed(123)
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )
    print("\nOutput text:\n", token_ids_to_text(token_ids, tokenizer)) 

if __name__ == "__main__":
    main()
    
# Next: fine-tuning for classification.py