class Config:
    import torch
    batches_to_train = 16
    epochs = 50
    vocab_size = 50257
    lr = 3e-4
    wd = 1e-5
    bs = 128
    n_embed = 768
    num_blocks = 12
    num_heads = 12
    head_size = n_embed // num_heads
    context_len = 224
    attn_drop_value = 0.2
    multihead_drop_value = 0.2
    ffn_drop_value = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    wandb = False