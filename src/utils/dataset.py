import torch
from torch.utils.data import Dataset
import os
import numpy as np
import configs

from utils import augmentation as aug
import random


def Catergorical2OneHotCoding(a, num_class=None):
    # a: [1, 4, 0, 5, 2]
    # type: numpy array
    if num_class:
        b = np.zeros((a.size, num_class))
    else:
        b = np.zeros((a.size, np.max(a) + 1))
    b[np.arange(a.size), a] = 1
    return b

class SupervisedDataset(Dataset):
    def __init__(self, mode):
        super().__init__()
        
        if mode == "train":
            self.ecg = np.load("/rdf/data/IMEC/semi_supervided/segments/window_{}_min/ecg_train.npy".format(configs.supResolution))
            self.gsr = np.load("/rdf/data/IMEC/semi_supervided/segments/window_{}_min/gsr_train.npy".format(configs.supResolution))
            self.label = np.load("/rdf/data/IMEC/semi_supervided/segments/window_{}_min/label_train.npy".format(configs.supResolution))
        elif mode == "test":
            self.ecg = np.load("/rdf/data/IMEC/semi_supervided/segments/ecg_test_.npy")
            self.gsr = np.load("/rdf/data/IMEC/semi_supervided/segments/gsr_test_.npy")
            self.label = np.load("/rdf/data/IMEC/semi_supervided/segments/label_test_.npy")
        else:
            print("Incorrect mode")
            
        self.label = self.label - 1
        self.label = self.label.astype(int)
        self.label[self.label > 0] = 1
        self.label = Catergorical2OneHotCoding(self.label)
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        ecg = torch.Tensor(self.ecg[idx].reshape((1,-1)))
        gsr = torch.Tensor(self.gsr[idx].reshape((1,-1)))
        label = torch.Tensor(self.label[idx])
        return ecg, gsr, label
    
class UnlabeledDataset(Dataset):
    def __init__(self, volume=1):
        super().__init__()
        
        self.ecg = np.load('/rdf/data/IMEC/semi_supervided/segments/all_segments/ecg_train.npy', mmap_mode='r')
        self.gsr = np.load('/rdf/data/IMEC/semi_supervided/segments/all_segments/gsr_train.npy', mmap_mode='r')
        
        if volume < 1:
            likelihood = np.load("/rdf/data/IMEC/semi_supervided/segments/all_segments/gmm_likelihood.npy")
            qualified_likelihood = np.percentile(likelihood, (1-volume) * 100)
            qualified_index = (likelihood >= qualified_likelihood)
            self.ecg = np.array(self.ecg[qualified_index])
            self.gsr = np.array(self.gsr[qualified_index])
        else:
            self.ecg = np.array(self.ecg)
            self.gsr = np.array(self.gsr)
        
    def __len__(self):
        return len(self.ecg)
    
    def normalization(self, x):
        # x = x.reshape((1,-1))
        x = (x - np.min(x, axis=1, keepdims=True))/(np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True) + 0.00000001)
        return x
    
    def transformation(self, seq):
        x = seq.copy()
        x = x.T
        args = random.choice(['jitter', 'scaling', 'permutation', 'magnitudewarp', 'timewarp', 'original'])
        if args == 'jitter':
            x = aug.jitter(x, sigma=configs.noise_sigma)
        elif args == 'scaling':
            x = aug.scaling(x, sigma=configs.noise_sigma)
        elif args == 'permutation':
            x = aug.permutation(x)
        elif args == 'magwarp':
            x = aug.magnitude_warp(x, sigma=configs.warp_sigma)
        elif args == 'timewarp':
            x = aug.time_warp(x, sigma=configs.warp_sigma)
        elif args == 'windowslice':
            x = aug.window_slice(x)
        elif args == 'windowwarp':
            x = aug.window_warp(x)
        else:
            pass;
        x = x.T
        x = self.normalization(x)
        return x
        
    
    def __getitem__(self, idx):
        ecg = self.ecg[idx].reshape((1,-1))
        gsr = self.gsr[idx].reshape((1,-1))
        
        ecg_1, ecg_2 = self.transformation(ecg), self.transformation(ecg)
        gsr_1, gsr_2 = self.transformation(gsr), self.transformation(gsr)
        
        ecg_1, ecg_2, gsr_1, gsr_2 = [torch.Tensor(x) for x in [ecg_1, ecg_2, gsr_1, gsr_2 ]]
        
        return ecg_1, ecg_2, gsr_1, gsr_2 
    

class PureSegmentDataSet_GSR(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, supervised=False, volume=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if supervised:
            self.X = np.load('/rdf/data/IMEC/semi_supervided/segments/window_5_min/gsr_train.npy').reshape(-1, 3840)
            if volume < 1:
                likelihood = np.load("/rdf/data/IMEC/semi_supervided/segments/all_segments/gmm_likelihood.npy")
                qualified_likelihood = np.percentile(likelihood, (1-volume) * 100)
                qualified_index = (likelihood >= qualified_likelihood)
                self.X = np.array(self.X[qualified_index])
        else:
            self.X = np.load('/rdf/data/IMEC/semi_supervided/segments/all_segments/gsr_train.npy').reshape(-1, 3840)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_train_tensor = self.X[idx].reshape(1,-1)
        # X_train_tensor = (X_train_tensor - np.min(X_train_tensor, axis=1, keepdims=True))/(np.max(X_train_tensor, axis=1, keepdims=True) - np.min(X_train_tensor, axis=1, keepdims=True) + 0.00000001)
        X_train_tensor = torch.Tensor(X_train_tensor)
        # print(X_train_tensor.size(), self.y_train[idx])
        return X_train_tensor
    
    
class PureSegmentDataSet_ECG(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, supervised=False, volume=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if supervised:
            self.X = np.load('/rdf/data/IMEC/semi_supervided/segments/window_5_min/ecg_train.npy')
            if volume < 1:
                likelihood = np.load("/rdf/data/IMEC/semi_supervided/segments/all_segments/gmm_likelihood.npy")
                qualified_likelihood = np.percentile(likelihood, (1-volume) * 100)
                qualified_index = (likelihood >= qualified_likelihood)
                self.X = np.array(self.X[qualified_index])
        else:
            self.X = np.load('/rdf/data/IMEC/semi_supervided/segments/all_segments/ecg_train.npy')
        # threshold = np.percentile(self.X, 99)
        # any value greater than threshold will be set as threshold.
        # self.X[self.X > threshold] = threshold

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        X_train_tensor = self.X[idx].reshape(1,-1)
        X_train_tensor = (X_train_tensor - np.min(X_train_tensor, axis=1, keepdims=True))/(np.max(X_train_tensor, axis=1, keepdims=True) - np.min(X_train_tensor, axis=1, keepdims=True) + 0.00000001)
        X_train_tensor = torch.Tensor(X_train_tensor)
        # print(X_train_tensor.size(), self.y_train[idx])
        return X_train_tensor