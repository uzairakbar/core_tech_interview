import numpy as np
from utils import (
    set_seed,
    sweep_plot,
)
from scipy.stats import lognorm

from optimizer import hyper_param_opt
from dataset import train_val_test_split, download_data
from graph import generate_graph


ALL_GRAPHS = {
    "Baseline": lambda indices, f: generate_graph(),
    "Noise>P": lambda indices, f: generate_graph(noise_perturbes=["paper"], indices=indices, f=f),
    "Noise>A": lambda indices, f: generate_graph(noise_perturbes=["author"], indices=indices, f=f),
    "Noise>S": lambda indices, f: generate_graph(noise_perturbes=["subject"], indices=indices, f=f),
    "Noise>PA": lambda indices, f: generate_graph(noise_perturbes=["paper", "author"], indices=indices, f=f),
    "Noise>PS": lambda indices, f: generate_graph(noise_perturbes=["paper", "subject"], indices=indices, f=f),
    "Noise>PAS": lambda indices, f: generate_graph(noise_perturbes=["paper", "author", "subject"], indices=indices, f=f),
}


SEARCH_SPACE = {
       'num_hidden': [2, 3],
       'hidden_size': [10,],
       'lr': lognorm(s=0.01, scale=0.01),
       'weight_decay': lognorm(s=0.01, scale=0.01),
       'epochs': [250,]
}


class Experiment():
    def __init__(self,
                 n_experiments = 10,
                 seed = 42,
                 graphs = "all",
                 sweep_samples = 10):
        self.n_experiments = n_experiments
        self.seed = seed
        self.sweep_samples = sweep_samples
        if graphs == "all":
            self.graphs = ALL_GRAPHS
        else:
            self.graphs = {m: ALL_GRAPHS[m] for m in graphs.split(',')}
        download_data()
    
    def fit(self, G, train_idx, val_idx, test_idx, labels):
        model, accuracy = hyper_param_opt(
            G, SEARCH_SPACE, train_idx, val_idx, test_idx, labels, num_trials=10
        )
        return model, accuracy
    
    def generate_dataset_split(self):
        train_idx, val_idx, test_idx, labels = train_val_test_split()
        return train_idx, val_idx, test_idx, labels

    def param_sweep(self):
        f_values = np.linspace(
            0, 1, num=self.sweep_samples
        )
        return f_values

    def run(self):
        if self.seed >= 0:
            set_seed(self.seed)
        param_values = self.param_sweep()
        
        error_dim = (self.sweep_samples, self.n_experiments)
        results = {name: np.zeros(error_dim) for name in self.graphs}
        
        total_exp = param_values*self.n_experiments*len(self.graphs)
        for i, param in enumerate(param_values):
            for j in range(self.n_experiments):
                train_idx, val_idx, test_idx, labels = self.generate_dataset_split()
                for k, (graph_name, graph) in enumerate(self.graphs.items()):
                    exp_num = (i+1)*(j+1)*(k+1)
                    print(f"EXPERIMENT {exp_num}/{total_exp} -- nth Noise: {i+1}/{self.sweep_samples}, Exp: {j+1}/{self.n_experiments}, Graph: {graph_name}")
                    G = graph(indices=(train_idx, val_idx, test_idx), f=param)
                    _, results[graph_name][i][j] = self.fit(
                        G, train_idx, val_idx, test_idx, labels
                    )
        return param_values, results
    



def main():
    f_values, results = Experiment().run()
    sweep_plot(
        f_values, results, xlabel=r"Noise Strength $f$", xscale="linear"
    )


if __name__ == "__main__":
    main()

