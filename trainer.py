import copy
import torch
from model import HeteroRGCN
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import class_weights

def trainer(
    G, train_idx, val_idx, test_idx, labels,
    hidden_size, num_hidden, lr, weight_decay, epochs, 
    verbose=True, plot_loss=False):
    model = HeteroRGCN(
        G, 
        in_size=10, hidden_size=hidden_size, out_size=3, 
        num_hidden=num_hidden,
    )
    opt = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    weights = class_weights(labels[train_idx])

    log_norm = []
    log_val_loss = []
    log_train_loss = []
    best_val_acc = 0
    best_test_acc = 0
    best_train_acc = 0
    best_val_loss = float("inf")
    for epoch in range(epochs):
        logits = model(G)
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(
            logits[train_idx], labels[train_idx], 
            weight=weights
        )
        val_loss = F.cross_entropy(
            logits[val_idx], labels[val_idx], 
            weight=weights
        ).item()
        
        pred = logits.argmax(1)
        train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
        val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
        test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

        norm = 0
        for n, p in model.named_parameters():
            if "perturbes" in n and "bias" not in n:
                norm += p.norm().item()

        if best_val_loss > val_loss:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_val_loss = val_loss
            best_train_acc = train_acc
            best_model = copy.deepcopy(model)
            best_norm = norm
            best_epoch = epoch

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % (epochs//10) == 0 and verbose:
            print('Epoch %d, Train Loss %.4f, Train Acc %.4f, Val Loss %.4f (Best %.4f), Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f), Norm %.4f (Best %.4f)' % (
                epoch,
                loss.item(),
                train_acc.item(),
                val_loss,
                best_val_loss,
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),
                norm,
                best_norm
            ))
        
        log_norm.append(norm)
        log_train_loss.append(loss.item())
        log_val_loss.append(val_loss)
    
    if plot_loss:
        plt.plot(log_train_loss)
        plt.plot(log_val_loss)
        plt.axvline(best_epoch, color='r')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["train", "val", "best"])
        plt.title(f"Training Config: #h-layers: {num_hidden}, #h-features: {hidden_size}, lr: {lr:.4f}, weight_decay: {weight_decay:.4f}")
        plt.grid()

    print(f"""TRAINING RESULT -- Train Acc: {best_train_acc:.4f}, Val Acc: {best_val_acc:.4f}, Test Acc: {best_test_acc:.4f}.\n""")
    return best_model.eval(), best_train_acc, best_val_acc, best_test_acc, best_val_loss