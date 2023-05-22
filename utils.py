import os
import dgl
import torch
import random
import numpy as np

import os
import torch
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def set_seed(seed = 42):
    np.random.seed(seed)
    
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)

    dgl.seed(seed)
    
    print(f"Random seed set as {seed}")


def sweep_plot(x, y,
               xlabel,
               ylabel="Test Accuracy",
               xscale="linear",
               vertical_plots=[],
               save=True):
    sns.set_style("darkgrid")
    colors = sns.color_palette()[:len(y)+1]
    fig = plt.figure()
    labels = []
    for i, (method, errors) in enumerate(y.items()):
        mean = errors.mean(axis = 1)

        label = method
        labels.append(label)

        if method in vertical_plots:
            plt.axvline(x = mean.mean(), color=colors[i], label=label)
        else:
            plt.plot(x, mean, color=colors[i], label=label)
        
    for i, (method, errors) in enumerate(y.items()):
        if method not in vertical_plots:
            low = np.percentile(errors, 2.5, axis=1)
            high = np.percentile(errors, 97.5, axis=1)
            plt.fill_between(x, low, high, color=colors[i], alpha = 0.1)
    
    plt.xlabel(xlabel), plt.ylabel(ylabel)
    plt.xlim([min(x), max(x)])
    plt.xscale(xscale)
    plt.legend(labels)
    plt.tight_layout()
    plt.show()
    if save:
        fname = "".join(c for c in xlabel if c.isalnum()) + "_sweep"
        fig.savefig(f"assets/{fname}.pdf", format="pdf", dpi=1200)

def class_weights(labels):
    classes_labels, counts = np.unique(labels, return_counts=True)
    class_weights = torch.tensor(max(counts)/counts).float()
    return class_weights