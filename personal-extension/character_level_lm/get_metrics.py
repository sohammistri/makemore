import os
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.optim as optim
from utils import create_data, compute_loss
from config_manager import ConfigManager
from models import RNNLM, LSTMLM, GRULM
from torch.utils.data import DataLoader

def get_metrics(file_path, checkpoint_path):
    # read the names
    names = []
    with open(file_path, "r") as f:
        for name in f.readlines():
            names.append(name.strip())

    # read pretrained model and get the mappings as well
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Extract config
    config = checkpoint['config']  # model configuration dict
    # Extract vocabulary mappings
    stoi = checkpoint['stoi']  # string to index mapping
    itos = checkpoint['itos']  # index to string mapping

    # split the names in train, val and test
    random.seed(42)
    names_shuffled = names.copy()
    random.shuffle(names_shuffled)

    n = len(names_shuffled)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_names = names_shuffled[:n_train]
    val_names = names_shuffled[n_train:n_train + n_val]
    test_names = names_shuffled[n_train + n_val:]

    # generate the train, val and test dataset
    train_x, train_y = create_data(train_names, stoi, padding=True,\
                                    max_seq_len=config["training"]["sequence_length"])
    val_x, val_y = create_data(val_names, stoi, padding=True,\
                                    max_seq_len=config["training"]["sequence_length"])
    test_x, test_y = create_data(test_names, stoi, padding=True,\
                                    max_seq_len=config["training"]["sequence_length"])

    train_x, train_y = torch.LongTensor(train_x), torch.LongTensor(train_y)
    val_x, val_y = torch.LongTensor(val_x), torch.LongTensor(val_y)
    test_x, test_y = torch.LongTensor(test_x), torch.LongTensor(test_y)

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True, drop_last=False)

    # create the model
    model_type = config["model"]["architecture"]["type"]
    device = config["hardware"]["device"]
    embedding_dim = config["model"]["architecture"]["embedding_dim"]
    hidden_dim = config["model"]["architecture"]["hidden_dim"]
    num_layers = config["model"]["architecture"]["num_layers"]
    dropout = config["model"]["architecture"]["dropout"]
    batch_first = config["model"]["architecture"]["batch_first"]

    if model_type == "rnn":
        model = RNNLM(vocab_size=len(stoi), embedding_dim=embedding_dim,\
                       hidden_dim=hidden_dim, num_layers=num_layers,\
                        dropout=dropout, batch_first=batch_first, device=device)

    elif model_type == "lstm":
        model = LSTMLM(vocab_size=len(stoi), embedding_dim=embedding_dim,\
                       hidden_dim=hidden_dim, num_layers=num_layers,\
                        dropout=dropout, batch_first=batch_first, device=device)

    elif model_type == "gru":
        model = GRULM(vocab_size=len(stoi), embedding_dim=embedding_dim,\
                       hidden_dim=hidden_dim, num_layers=num_layers,\
                        dropout=dropout, batch_first=batch_first, device=device)

    # load pretrain model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])

    # get the loss
    train_loss = compute_loss(dataloader=train_dataloader, model=model,\
                              criterion=criterion, ignore_idx=stoi["<pad>"], device=device)
    val_loss = compute_loss(dataloader=val_dataloader, model=model,\
                              criterion=criterion, ignore_idx=stoi["<pad>"], device=device)
    test_loss = compute_loss(dataloader=test_dataloader, model=model,\
                              criterion=criterion, ignore_idx=stoi["<pad>"], device=device)


    print(f"""| Train Loss | Val Loss | Test Loss |
|------------|------------|------------|
| {train_loss:.4f} | {val_loss:.4f} | {test_loss:.4f} |""")

if __name__ == "__main__":
    file_path = "/root/makemore/names.txt"
    checkpoint_path = "/root/makemore/personal-extension/character_level_lm/ckpt/lstm_best_model_smaller_vocab_finetune.pth"

    get_metrics(file_path, checkpoint_path)