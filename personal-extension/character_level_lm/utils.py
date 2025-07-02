from torch.utils.data import DataLoader
import wandb
import string
import numpy as np
import re

def create_pretrain_vocab_from_file():
    vocab = set()

    # Ensure all lowercase letters a-z are included
    vocab.update(string.ascii_lowercase)

    # Ensure '.' and '<pad>' tokens are present
    vocab.add(".")
    vocab.add("<pad>")
    vocab.add(" ")
    vocab.add("\n")

    # Sort vocab and create mappings
    vocab = sorted(vocab)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    return vocab, stoi, itos

def clean_file(input_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        content = infile.read()
    
    # Convert to lowercase
    content = content.lower()
    
    # Retain only lowercase letters, dots, spaces, and newlines
    cleaned = re.sub(r'[^a-z.\n ]', '', content)
    return cleaned

def create_pretrain_data_from_file(file_path, stoi, sequence_length):
    cleaned_content = clean_file(file_path)

    list_content = np.array([stoi[l] for l in cleaned_content], dtype=np.int32)

    # Total number of sequences we can extract
    num_sequences = len(list_content) - sequence_length

    # Preallocate arrays
    x = np.lib.stride_tricks.sliding_window_view(list_content, window_shape=sequence_length)[:num_sequences]
    y = np.lib.stride_tricks.sliding_window_view(list_content[1:], window_shape=sequence_length)[:num_sequences]

    return x, y

def create_and_save_pretrain_data(file_path, stoi, sequence_length, pretrain_file_path):
    x, y = create_pretrain_data_from_file(file_path, stoi, sequence_length)

    # split data
    n = len(x)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    x_train, y_train = x[:n_train, :], y[:n_train, :]
    x_val, y_val = x[n_train:n_train + n_val, :], y[n_train:n_train + n_val, :]
    x_test, y_test = x[n_train + n_val:, :], y[n_train + n_val:, :]

    # save the data
    np.savez(pretrain_file_path, x_train=x_train, y_train=y_train,\
             x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)

def create_vocab(names):
    vocab = sorted(set([l for name in names for l in name]))
    vocab.append(".")
    vocab.append("<pad>")

    stoi, itos = {}, {}

    for i, l in enumerate(vocab):
        stoi[l] = i 
        itos[i] = l

    return vocab, stoi, itos

def create_data(names, stoi, padding=False, max_seq_len=None):
    # create dataset like
    # for name soham -> make it .soham. -> [.,s,o,h,a,m] ---> [s,o,h,a,m,.]

    x, y = [], []

    for name in names:
        name = "." + name + "."
        tx, ty = [], []
        for l1, l2 in zip(name, name[1:]):
            tx.append(stoi[l1])
            ty.append(stoi[l2])

        if padding:
            assert max_seq_len is not None
            if len(tx) < max_seq_len:
                tx.extend([stoi["<pad>"]] * (max_seq_len - len(tx)))
                ty.extend([stoi["<pad>"]] * (max_seq_len - len(ty)))

        x.append(tx)
        y.append(ty)

    return x, y

def compute_loss(dataloader, model, criterion, ignore_idx, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    for x, y in dataloader:
        x = x.to(device) # (B, seq_len)
        y = y.to(device) # (B, seq_len)

        logits = model(x) # (B, C, V)
        B, C, V = logits.shape
        loss = criterion(logits.view(B * C, V), y.view(B * C))

        valid_mask =  (y.view(B*C) != ignore_idx)
        num_valid = valid_mask.sum().item()

        total_loss += loss.item() * num_valid
        total_count += num_valid

    return total_loss / total_count if total_count > 0 else float('nan')

def train(train_dataset, val_dataset, batch_size, n_eopchs, model, optimizer, criterion, ignore_idx, config, device):
    wandb.init(
        project=config.get("model.name"),   # Change this
        config=config.to_dict()
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    val_loss = compute_loss(val_dataloader, model, criterion, ignore_idx, device)

    wandb.log({
        "val_loss": val_loss
    })
    # print(f"Start of training: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    for epoch in range(n_eopchs):
        # train loop
        model.train()
        i = 0
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            B, C, V = logits.shape
            loss = criterion(logits.view(B * C, V), y.view(B * C))
            loss.backward()
            optimizer.step()
            wandb.log({
                "train_loss": loss.item(),
            })
            i += 1
            if i % 50 == 0:
                model.eval()
                val_loss = compute_loss(val_dataloader, model, criterion, ignore_idx, device)
                wandb.log({
                    "val_loss": val_loss
                })
                model.train()

    wandb.finish()