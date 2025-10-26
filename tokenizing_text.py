from importlib.metadata import version
# Check the versions of the libraries
print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

# Get the text data
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
    
# TOKENIZING TEXT    

print("Total number of character:", len(raw_text))
print(raw_text[:99])

import re
# Tokenize the text into sentences
text = "Hello world. This is a test. Let's see how it tokenizes."
result = re.split(r'(\s)', text)
print(result)

# Modify the regex pattern to split punctuation
'''result = re.split(r'([,.]|\s)', text)
print(result)'''

# Remove empty strings (tokens)/whitespaces
# Strip whitespace from each item and then filter out any empty strings.
'''result = [item for item in result if item.strip()]
print(result) '''

# Modify the regex pattern to split other types of punctuation
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

# Tokenize the entire text data
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("Total number of tokens:", len(preprocessed))

# Show the first 30 tokens
print(preprocessed[:30])


# CONVERTING TOKENS INTO TOKEN IDS

# Create a list of all unique tokens and sort them alphabetically
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("Vocab size:", vocab_size)

# Creating a vocabulary
vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50 : # print the vocabulary's first 51 items for illustration
        break

# Implementing a simple text tokenizer
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab # stores the vocabulary as a class attribute for access in the encode and decode methods
        self.int_to_str = {i:s for s, i in vocab.items()} # creates an inverse vocabulary that maps token IDs back to the original text tokens
        
    def encode(self, text): # processes input text into token IDs
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids): # converts token IDs back into text
        text = " ".join([self.int_to_str[i] for i in ids])
        
        text = re.sub(r'\s+([,.!?"()\'])', r'\1', text) # removes spaces before the specified punctuation
        return text

# Instantiate a new tokenizer object from the SimpleTokenizerV1 class and tokenize a sample text from short story   
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print("Token IDs:", ids)      

# Turn the token IDs back into text
print("Decoded text:", tokenizer.decode(ids))

# Apply it to a new text sample not contained in the training set
# text = "Hello, do you like tea?"
# print("New text:", tokenizer.encode(text))
# This will raise a KeyError because the tokenizer's vocabulary does not include the word Hello since it was not used in The Verdict.
# This shows the need to consider large and diverse training sets to extend the vocabulary when working on LLMs.

# ADDING SPECIAL CONTEXT TOKENS

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"]) # add special tokens to the vocabulary
vocab = {token:integer for integer, token in enumerate(all_tokens)}
print("New vocab size:", len(vocab.items()))

# Show the last 5 items of the updated vocabulary as a quick check
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# A simple text tokenizer that handles unknown words
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
        
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int # replaces unknown words by the <|unk|> tokens
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.!?"()\'])', r'\1', text) # replaces spaces before the specified punctuations
        return text   

# Try the new tokenizer out in practice
text1 = "Hello, do you like ice tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print("New text:", text)    

tokenizer = SimpleTokenizerV2(vocab)
print("Token IDs:", tokenizer.encode(text))

print("Decoded text:", tokenizer.decode(tokenizer.encode(text)))

# BYTE PAIR ENCODING (BPE)

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

# NEXT: data sampling with a sliding window (continued in data-sampling_w_sliding_window.py with cleaned up code)