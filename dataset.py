import torch
import scipy.io
import numpy as np
import urllib.request

def download_data():
    data_url = 'https://data.dgl.ai/dataset/ACM.mat'
    data_file_path = '/content/ACM.mat'
    urllib.request.urlretrieve(data_url, data_file_path)

def train_val_test_split():
    data_file_path = '/content/ACM.mat'
    data = scipy.io.loadmat(data_file_path)
    print(list(data.keys()))

    num_papers = data['PvsA'].shape[0]
    num_authors = data['PvsA'].shape[1]
    num_subjects = data['PvsA'].nnz
    num_noise = num_papers + num_authors + num_subjects

    pvc = data['PvsC'].tocsr()
    # find all papers published in KDD, ICML, VLDB
    c_selected = [0, 11, 13]  # KDD, ICML, VLDB
    p_selected = pvc[:, c_selected].tocoo()
    # generate labels
    labels = pvc.indices
    labels[labels == 11] = 1
    labels[labels == 13] = 2
    labels = torch.tensor(labels).long()

    # generate train/val/test split
    pid = p_selected.row
    print(len(pid))
    shuffle = np.random.permutation(pid)
    train_idx = torch.tensor(shuffle[0:800]).long()
    val_idx = torch.tensor(shuffle[800:1000]).long()
    test_idx = torch.tensor(shuffle[1000:]).long()

    return train_idx, val_idx, test_idx, labels