from torch.utils.data import DataLoader
import wandb

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

    train_loss_dict, val_loss_dict = {}, {}

    train_loss = compute_loss(train_dataloader, model, criterion, ignore_idx, device)
    val_loss = compute_loss(val_dataloader, model, criterion, ignore_idx, device)

    wandb.log({
        "epoch": 0,
        "train_loss": train_loss,
        "val_loss": val_loss
    })
    # print(f"Start of training: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    for epoch in range(n_eopchs):
        # train loop
        model.train()
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            B, C, V = logits.shape
            loss = criterion(logits.view(B * C, V), y.view(B * C))
            loss.backward()
            optimizer.step()

        # Compute loss
        train_loss = compute_loss(train_dataloader, model, criterion, ignore_idx, device)
        val_loss = compute_loss(val_dataloader, model, criterion, ignore_idx, device)

        train_loss_dict[epoch + 1] = train_loss
        val_loss_dict[epoch + 1] = val_loss

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

    wandb.finish()
    return train_loss_dict, val_loss_dict