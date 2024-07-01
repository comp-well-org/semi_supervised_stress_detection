from model_rl.autoencoder import CNN_AE_GSR, CNN_AE_ECG, model_conv1d_autoencoder
from torchinfo import summary
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from utils.dataset import PureSegmentDataSet_ECG, PureSegmentDataSet_GSR
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--modality', type=str, default='ecg', help='modality to train')
parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--dataset', type=str, default='SMILE', help='dataset to train')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
parser.add_argument('--supervised', type=bool, default=True, help='supervised or not')
parser.add_argument('--unlabel_volume', type=float, default=0.4, help='volume (percentage) of unlabeled data')
parser.add_argument('--model_save_path', type=str, default='./results/representation_learning/SMILE_ECG/', help='path to save model')
args = parser.parse_args()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def create_segments_loader(source):
    if source == "gsr":
        dataset = PureSegmentDataSet_GSR(args.supervised, volume=args.unlabel_volume)
    elif source == "ecg":
        dataset = PureSegmentDataSet_ECG(args.supervised, volume=args.unlabel_volume)
        
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)
    return loader

def train(loader, model, device):
    optimizer_encoder = torch.optim.Adam(model.parameters(), lr=args.lr)
    # lr_scheduler = StepLR(optimizer_encoder, step_size=30, gamma=0.1)
    tb_writer = SummaryWriter(log_dir = args.model_save_path, comment='init_run')
    loss_func = nn.MSELoss()
    for e in range(args.epochs):
        model.train()
        epoch_loss = 0
        for x in tqdm(loader):
            x_, _ = model(x.to(device))
            loss = loss_func(x.to(device), x_)
            
            optimizer_encoder.zero_grad()
            loss.backward()
            epoch_loss += loss.item() / len(loader)
            optimizer_encoder.step()
        # lr_scheduler.step()
        tb_writer.add_scalar('Recon Loss/Train', epoch_loss, e)
        print('Training Epoch {} - Recon Loss : {}'.format(e, epoch_loss))
        if (e + 1) % 5 == 0:
            save_path = args.model_save_path + 'supervised_{}_checkpoint_{}.pth'.format(int(args.supervised),e + 1)
            torch.save(model.state_dict(), save_path)

def infer(loader, model, device):
    model.eval()
    res_infer = []
    for x in tqdm(loader):
        x_, feat = model(x.to(device))
        res_infer.append(feat.detach().cpu().squeeze(-1).numpy())
    feat_numpy = np.concatenate(res_infer, axis=0)
    np.save("labeled_feats_{}.npy".format(args.modality), feat_numpy)
    print(1)

def main():
    dataloader = create_segments_loader(args.modality)
    model = model_conv1d_autoencoder(1, modality=args.modality).to(device)
    if args.mode == "train":
        train(dataloader, model, device)
    elif args.mode == "infer":
        model.load_state_dict(torch.load("/home/hy29/rdf/semi_supervised_v2/results/representation_learning/ecg_conv_ae/checkpoint_40.pth"))
        infer(dataloader, model, device)
    
if __name__ == '__main__':
    main()

