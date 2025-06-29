import os
import torch
from typing import Dict, Any
from models import RNNLM, LSTMLM, GRULM

def generate_samples(model, start_tensor, itos, sequence_length, result_path):
    B = start_tensor.shape[0] # B
    x = start_tensor.view(B, 1)

    for i in range(sequence_length):
        logits = model(x) # (B, seq_len, V)
        target_logits = logits[:, -1, :] # (B, V)
        probs = torch.softmax(target_logits, dim=1)
        sampled_indices = torch.multinomial(probs, num_samples=1) # (B, 1)
        x = torch.cat([x, sampled_indices], dim=1) # (B, seq_len + 1)

    res = x.clone() # (B, seq_len)
    res_list = res.cpu().detach().tolist()

    generated_names = []
    for res in res_list:
        dot_count = 0
        name = ""
        for idx in res:
            l = itos[idx]
            name += l
            if l == ".":
                dot_count += 1
            if dot_count == 2:
                break
        
        name = name.strip(".")
        generated_names.append(name)
    
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        for name in generated_names:
            f.write(name+"\n")

def main(ckpt_path, result_path, num_samples=10):
    # load ckpt
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # Extract config
    config = checkpoint['config']  # model configuration dict
    # Extract vocabulary mappings
    stoi = checkpoint['stoi']  # string to index mapping
    itos = checkpoint['itos']  # index to string mapping

    # initialize the model
    model_type = config["model"]["architecture"]["type"]
    device = config["hardware"]["device"]
    embedding_dim = config["model"]["architecture"]["embedding_dim"]
    hidden_dim = config["model"]["architecture"]["hidden_dim"]
    num_layers = config["model"]["architecture"]["num_layers"]
    dropout = config["model"]["architecture"]["dropout"]
    batch_first = config["model"]["architecture"]["batch_first"]

    print(model_type)

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

    # load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # move to gpu
    model.to(device)

    # prepare for decoding
    start_idx = stoi["."]
    start_tensor = torch.LongTensor([start_idx] * num_samples).to(device) # (B,)
    generate_samples(model, start_tensor, itos, config["training"]["sequence_length"], result_path)


if __name__ == "__main__":
    ckpt_path = "/root/makemore/personal-extension/character_level_lm/ckpt/gru_best_model.pth"
    result_path = "results/gru_scratch.txt"
    num_samples = 50
    main(ckpt_path, result_path, num_samples)