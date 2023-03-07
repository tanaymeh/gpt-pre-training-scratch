import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from transformers import GPT2TokenizerFast

class CustomRandomSampler(Sampler):
    def __init__(self, data, context_length, replacement=False, num_samples=None):
        self.data = data
        self.context_length = context_length
        self.replacement = replacement
        self._num_samples = num_samples
        
    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data)
        return self._num_samples

    def __iter__(self):
        n = len(self.data) - self.context_length
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples
    
class TextDataset(Dataset):
    def __init__(self, text_file, config, tokenizer=None):
        self.text_file = text_file.split()
        self.file_size = len(self.text_file)
        self.config = config
        self.tokenizer = tokenizer if tokenizer else GPT2TokenizerFast.from_pretrained('gpt2')
        
    def __getitem__(self, idx):
        """
        Key idea here is to just tokenize the little window of text (from idx to idx+context_len + 1) that we will be using
        This way you are not tokenizing entire file everytime.
        """
        current_window = self.text_file[idx:idx+self.config.context_len+1]
        tokenized_output = self.tokenizer.encode(" ".join(current_window), return_tensors='pt')[0]
        text = tokenized_output[0:self.config.context_len]
        target = tokenized_output[1:self.config.context_len+1]
        return (text, target)

    def __len__(self):
        return self.file_size