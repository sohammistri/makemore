import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.optim as optim
from utils import create_pretrain_vocab_from_file, create_and_save_pretrain_data, train
from config_manager import ConfigManager
from models import RNNLM, LSTMLM, GRULM

def main(config, file_path):
    # generate vocab maps
    vocab, stoi, itos = create_pretrain_vocab_from_file()

    # generate data
    sequence_length = config.get("training.sequence_length", 64)
    pretrain_file_path = config.get("training.file_path")
    # create_and_save_pretrain_data(file_path, stoi, sequence_length, pretrain_file_path)

    data = np.load(pretrain_file_path)
    x_train, y_train, x_val, y_val, x_test, y_test = data["x_train"], data["y_train"],\
          data["x_val"], data["y_val"], data["x_test"], data["y_test"]

    train_x, train_y = torch.LongTensor(x_train), torch.LongTensor(y_train)
    val_x, val_y = torch.LongTensor(x_val), torch.LongTensor(y_val)
    test_x, test_y = torch.LongTensor(x_test), torch.LongTensor(y_test)

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    # create the model
    model_type = config.get("model.architecture.type")
    device = config.get("hardware.device")
    embedding_dim = config.get("model.architecture.embedding_dim")
    hidden_dim = config.get("model.architecture.hidden_dim")
    num_layers = config.get("model.architecture.num_layers")
    dropout = config.get("model.architecture.dropout")
    batch_first = config.get("model.architecture.batch_first")

    if model_type == "rnn":
        model = RNNLM(vocab_size=len(vocab), embedding_dim=embedding_dim,\
                       hidden_dim=hidden_dim, num_layers=num_layers,\
                        dropout=dropout, batch_first=batch_first, device=device)

    elif model_type == "lstm":
        model = LSTMLM(vocab_size=len(vocab), embedding_dim=embedding_dim,\
                       hidden_dim=hidden_dim, num_layers=num_layers,\
                        dropout=dropout, batch_first=batch_first, device=device)

    elif model_type == "gru":
        model = GRULM(vocab_size=len(vocab), embedding_dim=embedding_dim,\
                       hidden_dim=hidden_dim, num_layers=num_layers,\
                        dropout=dropout, batch_first=batch_first, device=device)

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
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stoi': stoi,
        'itos': itos,
        'config': config.to_dict() 
    }
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    file_path = "/root/makemore/personal-extension/character_level_lm/cleaned_merged_fairy_tales_without_eos.txt"
    config_path = "config_pretrain.yml"
    config = ConfigManager(config_path)

    main(config, file_path)