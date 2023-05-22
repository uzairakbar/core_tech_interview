import copy
from trainer import trainer
from sklearn.model_selection import ParameterSampler

def hyper_param_opt(
    G, search_space, train_idx, val_idx, test_idx, labels, num_trials=10, verbose=False
    ):
    config_generator = ParameterSampler(search_space, n_iter=num_trials)
    best_val_acc = 0
    best_test_acc = 0
    best_train_acc = 0
    best_val_loss = float("inf")
    for i, config in enumerate(config_generator):
        model, train_acc, val_acc, test_acc, val_loss = trainer(
            G=copy.deepcopy(G),
            train_idx=train_idx,
            val_idx=val_idx, 
            test_idx=test_idx,
            labels=labels,
            num_hidden=config['num_hidden'],
            hidden_size=config['hidden_size'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            epochs=config['epochs'],
            early_stopping=config['early_stopping'],
            verbose=verbose,
        )
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_config = config
            best_model = model
            best_val_loss = val_loss
            best_train_acc = train_acc
        
        print(
            f"""Trial: {i}, Config: {config},
            Val Loss: {val_loss:.4f} (Best {best_val_loss:.4f}), Val Acc: {val_acc:.4f} (Best {best_val_acc:.4f}), Test Acc: {test_acc:.4f} (Best {best_test_acc:.4f})\n"""
        )

    print(
        f"""\nBEST CONFIG: {best_config},
        Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}"""
    )
    return best_model, best_train_acc, best_val_acc, best_test_acc