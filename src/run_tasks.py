from pkgutil import get_data
import torch
from torch.utils.data import DataLoader
from utils.dataset import SupervisedDataset, UnlabeledDataset
from train import train, train_semi
from model.resnet import model_ResNet_SMILE
from model.cnn import model_conv1d

import argparse
import configs

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def get_data_loader():
    train_set, test_set = SupervisedDataset('train'), SupervisedDataset('test')
    train_loader = DataLoader(
        train_set,
        batch_size=configs.BATCH_SIZE,
        shuffle=True)
    test_loader = DataLoader(
        test_set,
        batch_size=configs.BATCH_SIZE // 4,
        shuffle=False)
    return train_loader, test_loader

def get_data_loader_unlabel():
    unlabeled_set = UnlabeledDataset(volume=0.1)
    unlabeled_loader = DataLoader(
        unlabeled_set,
        batch_size=configs.BATCH_SIZE_UNLABELED,
        shuffle=True)
    return unlabeled_loader


def main():
    train_loader, test_loader = get_data_loader()
    model = model_conv1d(load_pretrained=configs.load_pretrained)
    
    if configs.training_mode == "supervised":
        train(model, train_loader, test_loader, device)
    elif configs.training_mode == "semi_supervised":
        unlabeled_loader = get_data_loader_unlabel()
        train_semi(model, train_loader, unlabeled_loader, test_loader, device)
    
    

if __name__ == '__main__':
    main()
    