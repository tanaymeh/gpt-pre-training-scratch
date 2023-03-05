import os
import time
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader

import wandb
from transformers import GPT2TokenizerFast

from .config import Config
from .model import GPT, generate, loss_fn
from .process_data import TextDataset, CustomRandomSampler

def wandb_log(**kwargs):
    for k, v in kwargs.items():
        wandb.log({k:v})

def train_one_epoch(model, optimizer, dataloader, loss_fn, wandb=False):
    """
    Trains the model for one epoch
    """
    prog_bar = tqdm(dataloader, total=len(dataloader))
    for input_text, target_text in prog_bar:
        logits = model(input_text)
        loss = loss_fn(logits, target_text)
        loss_itm = loss.item()
        if wandb:
            wandb_log(loss=loss_itm)
        prog_bar.set_description(f"loss: {loss_itm:.4f}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def valid_one_epoch(model, dataloader, loss_fn, wandb=False):
    prog_bar = tqdm(dataloader, total=len(dataloader))
    for input_text, target_text in prog_bar:
        logits = model(input_text)
        loss = loss_fn(logits, target_text)
        loss_itm = loss.item()
        if wandb:
            wandb_log(loss=loss_itm)
        prog_bar.set_description(f"loss: {loss_itm:.4f}")


if __name__ == "__main__":
    config = Config()
    parent_path = "/kaggle/input/book-corpus-dataset-split"
    all_files = os.listdir("/kaggle/input/book-corpus-dataset-split")
    all_files = [os.path.join(parent_path, x) for x in all_files if x.endswith(".pt")]
    
    # Init config
    config = Config()

    model = GPT(config).to(config.device)
    if config.pretrained:
        model.load_state_dict(torch.load(config.pretrained))
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
    
    # Keep an eye on the model
    if config.wandb:
        wandb.watch(model)

    # Encode data
    start_time = time.time()
    fl = all_files[0]
    tokenized_file = torch.load(fl, map_location=config.device)
    for epx in range(config.epochs):
        print(f"{'='*40} Epoch: {epx+1}/{config.epochs} {'='*40}")
        dataset = TextDataset(config, tokenized_file)
        loader = DataLoader(
            dataset
            sampler=CustomRandomSampler(
                dataset
                config.context_len,
                replacement=True,
                num_samples=config.bs * config.batches_to_train
            )
        )
            
        print(f"Training on {os.path.basename(fl)}")

        # Train and Validate the model for 1 epochs each
        train_one_epoch(model, optim, loader, loss_fn=loss_fn, wandb=config.wandb)
        valid_one_epoch(model, loader, loss_fn, wandb=config.wandb)

    end = time.time() - start_time
    time_taken = end / (60 * 60)
    print(f"Took {time_taken:.1f} hours to train the model")
    
    # Save the model
    torch.save(model.state_dict(), config.model_name)

    # Generate some text
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    prompt = tokenizer.encode("In my opinion", return_tensors='pt').to(config.device)
    generated_text = generate(model, prompt, max_tokens=config.context_len)
    generated_text = tokenizer.decode(generated_text.tolist()[0])
    print(generated_text)

    if config.wandb:
        wandb.finish()