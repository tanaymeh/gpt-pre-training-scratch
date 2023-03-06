"""
Prepare data for training.
Basically take a big txt file with text and tokenize every 'k' lines of it
and store it as a separate pytorch '.pt' file
"""
import gc
import torch
from tqdm.auto import tqdm
from transformers import GPT2TokenizerFast

# For saving tokenized files
tok = GPT2TokenizerFast.from_pretrained("gpt2")
max_lines = 500_00_0 # Maximum number of lines in each file
max_files = 45 # Maximum number of files to be created
current_file = []
with open("/kaggle/input/english-bookcorpus/en.txt", "r") as file:
    for idx, line in tqdm(enumerate(file), total=max_lines*max_files):
        if max_files == 0:
            print("Done")
            break
            
        if idx == 0:
            current_file.append(line + "\n")
            continue
            
        if idx % max_lines == 0:
            current_file.append(line + "\n")
            joined_file = " ".join(current_file)
            # Tokenize the text
            encoding = tok.encode(joined_file, return_tensors='pt')[0]
            torch.save(encoding, f"file_split_{max_files}.pt")
            max_files -= 1
            current_file = []
            continue
            
        current_file.append(line + "\n")