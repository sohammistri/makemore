import os
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.optim as optim
from utils import create_vocab, create_data, train
from config_manager import ConfigManager
from models import RNNLM, LSTMLM, GRULM

def main(config, file_path):
    # read the names
    names = []
    with open(file_path, "r") as f:
        for name in f.readlines():
            names.append(name.strip())

    # read pretrained model and get the mappings as well
    pretrain_ckpt_path = config.get("model.pretrain_ckpt_path")
    pretrain_checkpoint = torch.load(pretrain_ckpt_path, map_location="cpu")
    # Extract config
    pretrain_config = pretrain_checkpoint['config']  # model configuration dict
    # Extract vocabulary mappings
    stoi = pretrain_checkpoint['stoi']  # string to index mapping
    itos = pretrain_checkpoint['itos']  # index to string mapping

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
                                    max_seq_len=config.get("training.sequence_length", 64))
    val_x, val_y = create_data(val_names, stoi, padding=True,\
                                    max_seq_len=config.get("training.sequence_length", 64))
    test_x, test_y = create_data(test_names, stoi, padding=True,\
                                    max_seq_len=config.get("training.sequence_length", 64))

    train_x, train_y = torch.LongTensor(train_x), torch.LongTensor(train_y)
    val_x, val_y = torch.LongTensor(val_x), torch.LongTensor(val_y)
    test_x, test_y = torch.LongTensor(test_x), torch.LongTensor(test_y)

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    # create the model
    model_type = pretrain_config["model"]["architecture"]["type"]
    device = pretrain_config["hardware"]["device"]
    embedding_dim = pretrain_config["model"]["architecture"]["embedding_dim"]
    hidden_dim = pretrain_config["model"]["architecture"]["hidden_dim"]
    num_layers = pretrain_config["model"]["architecture"]["num_layers"]
    dropout = pretrain_config["model"]["architecture"]["dropout"]
    batch_first = pretrain_config["model"]["architecture"]["batch_first"]

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
    model.load_state_dict(pretrain_checkpoint['model_state_dict'])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {total_params}")

    # train the model
    batch_size = config.get("training.batch_size")
    n_epochs = config.get("training.epochs")
    lr = config.get("training.learning_rate")
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])

    train(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=batch_size, n_eopchs=n_epochs,\
          model=model, optimizer=optimizer, criterion=criterion, config=config, ignore_idx=stoi["<pad>"], device=device)

    # save the ckpt
    os.makedirs("ckpt", exist_ok=True)
    checkpoint_path = os.path.join("ckpt", config.get("model.ckpt_name") + ".pth")
    config_dict = config.to_dict()
    config_dict.update(pretrain_config)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stoi': stoi,
        'itos': itos,
        'config': config_dict
    }
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    file_path = "/root/makemore/names.txt"
    config_path = "config_finetune.yml"
    config = ConfigManager(config_path)

    main(config, file_path)