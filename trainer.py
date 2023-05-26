import copy
import torch
from model import HeteroRGCN
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import class_weights

def trainer(
    G, train_idx, val_idx, test_idx, labels,
    hidden_size, num_hidden, lr, weight_decay, epochs, early_stopping=True,
    verbose=True, plot_loss=False, plot_norm=False):
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

        if best_val_loss > val_loss and early_stopping:
            # early-stopping: pick best model based on val_loss
            best_train_loss = loss.item()
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_val_loss = val_loss
            best_train_acc = train_acc
            best_model = copy.deepcopy(model)
            best_norm = norm
            best_epoch = epoch
        elif not early_stopping:
            best_train_loss = loss.item()
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_val_loss = val_loss
            best_train_acc = train_acc
            best_model = model
            best_norm = norm
            best_epoch = epoch

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % (epochs//10) == 0 and verbose:
            print(
                f"Epoch {epoch}, "
                f"Train Loss {loss.item():.4f} (Best {best_train_loss:.4f}), "
                f"Val Loss {val_loss:.4f} (Best {best_val_loss:.4f}), "
                f"Val Acc {val_acc:.4f} (Best {best_val_acc:.4f}), "
                f"Test Acc {test_acc:.4f} (Best {best_test_acc:.4f}), "
                f"Norm {norm:.4f} (Best {best_norm:.4f}), "
            )
        
        log_norm.append(norm)
        log_train_loss.append(loss.item())
        log_val_loss.append(val_loss)
    
    if plot_loss:
        plt.plot(log_train_loss)
        plt.plot(log_val_loss)
        plt.axvline(best_epoch, color='r')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["train", "val", "selected"])
        plt.title(
            f"Training Config: #h-layers: {num_hidden}, "
            f"#h-features: {hidden_size}, "
            f"lr: {lr:.4f}, "
            f"weight_decay: {weight_decay:.4f}"
        )
        plt.grid()

    if plot_norm:
        plt.plot(log_norm)
        plt.axvline(best_epoch, color='r')
        plt.xlabel("epoch")
        plt.ylabel("L2 Norm of Noise Perturbation Edge Weights")
        plt.legend(["norm", "selected"])
        plt.title(
            f"Training Config: #h-layers: {num_hidden}, "
            f"#h-features: {hidden_size}, "
            f"lr: {lr:.4f}, "
            f"weight_decay: {weight_decay:.4f}"
        )
        plt.grid()

    print(
        f"TRAINING RESULT -- "
        f"Train Acc: {best_train_acc:.4f}, "
        f"Val Acc: {best_val_acc:.4f}, "
        f"Test Acc: {best_test_acc:.4f}.\n"
    )
    return best_model.eval(), best_train_acc, best_val_acc, best_test_acc, best_val_loss