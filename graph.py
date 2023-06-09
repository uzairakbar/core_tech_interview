import scipy
import torch
import dgl

def generate_graph(noise_perturbes=None, f=0.0):
    data_file_path = '/content/ACM.mat'
    data = scipy.io.loadmat(data_file_path)
    num_papers = data['PvsA'].shape[0]
    num_authors = data['PvsA'].shape[1]
    num_subjects = data['PvsA'].nnz
    num_noise = num_papers + num_authors + num_subjects

    noise = torch.arange(num_noise)
    papers, paper_noise = torch.arange(num_papers), noise[:num_papers]
    authors, author_noise = torch.arange(num_authors), noise[num_papers:-num_subjects]
    subjects, subject_noise = torch.arange(num_subjects), noise[-num_subjects:]
    
    noise_targets = {
        "paper": (paper_noise, papers),
        "author": (author_noise, authors),
        "subject": (subject_noise, subjects),
    }
    
    data_dict = {
        ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
        ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
        ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
    }
    noise_perturbes = [] if noise_perturbes is None else noise_perturbes
    for target in noise_perturbes:
        data_dict[('noise', 'perturbes', target)] = noise_targets[target]

    num_nodes_dict = {
        'paper' : num_papers,
        'author' : num_authors,
        'subject' : num_subjects,
        'noise' : num_noise,
    }

    G = dgl.heterograph(
        data_dict = data_dict,
        num_nodes_dict = num_nodes_dict,
    )

    noise_features = f * torch.rand(num_noise, 1)
    G.nodes["noise"].data["h"] = noise_features
    
    return G