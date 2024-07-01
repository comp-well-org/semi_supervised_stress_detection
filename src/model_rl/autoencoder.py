import torch
import torch.nn as nn
from collections import OrderedDict

class CNN_AE_GSR(nn.Module):
    """Some Information about CNN_AE"""
    def __init__(self, in_channels):
        super(CNN_AE_GSR, self).__init__()
        self.encoder = self.create_encoder(in_channels=in_channels)
        self.decoder = self.create_embedding_recon()
        
    def create_encoder(self, in_channels):
        encoder = nn.Sequential(
            self.create_conv_block(in_channels, 8, 10, 3, 2),
            self.create_conv_block(8, 8, 10, 3, 2),
            self.create_conv_block(8, 16, 5, 2, 1),
            self.create_conv_block(16, 16, 5, 2, 1),
            self.create_conv_block(16, 32, 2, 2, 1),
            self.create_conv_block(32, 32, 2, 2, 1),
            self.create_conv_block(32, 64, 2, 2, 1),
            nn.AdaptiveAvgPool1d(1)
        )
        return encoder
        
    def create_conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm1d(out_channels),
            nn.ReLU()
        )
        return conv_block
    
    def create_embedding_recon(self):
        decoder = nn.Sequential(
            self._build_dec_block(64, 32, 2, stride=2, padding=0, output_padding=0),
            self._build_dec_block(32, 32, 2, stride=2, padding=0, output_padding=0),
            self._build_dec_block(32, 16, 2, stride=2, padding=0, output_padding=0),
            self._build_dec_block(16, 16, 5, stride=3, padding=1, output_padding=0),
            self._build_dec_block(16, 8, 6, stride=2, padding=1, output_padding=0),
            self._build_dec_block(8, 8, 8, stride=3, padding=0, output_padding=0),
            self._build_dec_block(8, 4, 10, stride=2, padding=0, output_padding=0),
            self._build_dec_block(4, 4, 10, stride=3, padding=2, output_padding=0),
            self._build_dec_block(4, 1, 10, stride=2, padding=1, output_padding=0),
        )
        return decoder
        
    def _build_dec_block(self, inplane, outplane, kernel_size, stride, padding, output_padding):
        block = nn.Sequential(
            nn.ConvTranspose1d(inplane, outplane, kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.BatchNorm1d(outplane),
            nn.ReLU()
        )
        return block
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
class CNN_AE_ECG(nn.Module):
    """Some Information about CNN_AE"""
    def __init__(self, in_channels):
        super(CNN_AE_ECG, self).__init__()
        self.encoder = self.create_encoder(in_channels=in_channels)
        self.decoder = self.create_embedding_recon()
        
    def create_encoder(self, in_channels):
        encoder = nn.Sequential(
            self.create_conv_block(in_channels, 8, 10, 3, 2),
            self.create_conv_block(8, 8, 10, 3, 2),
            self.create_conv_block(8, 16, 5, 2, 1),
            self.create_conv_block(16, 16, 5, 2, 1),
            self.create_conv_block(16, 32, 2, 2, 1),
            self.create_conv_block(32, 32, 2, 2, 1),
            self.create_conv_block(32, 64, 2, 2, 1),
            self.create_conv_block(64, 64, 2, 2, 1),
            self.create_conv_block(64, 128, 2, 2, 1),
            self.create_conv_block(128, 128, 2, 2, 1),
            self.create_conv_block(128, 256, 2, 2, 1),
            nn.AdaptiveAvgPool1d(1)
        )
        return encoder
        
    def create_conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        return conv_block
    
    def create_embedding_recon(self):
        decoder = nn.Sequential(
            self._build_dec_block(256, 128, 2, stride=2, padding=0, output_padding=0),
            self._build_dec_block(128, 64, 2, stride=3, padding=0, output_padding=0),
            self._build_dec_block(64, 32, 2, stride=2, padding=0, output_padding=0),
            self._build_dec_block(32, 32, 2, stride=2, padding=1, output_padding=0),
            self._build_dec_block(32, 16, 2, stride=2, padding=1, output_padding=0),
            self._build_dec_block(16, 16, 5, stride=3, padding=0, output_padding=0),
            self._build_dec_block(16, 8, 6, stride=2, padding=1, output_padding=0),
            self._build_dec_block(8, 8, 8, stride=3, padding=0, output_padding=0),
            self._build_dec_block(8, 4, 10, stride=2, padding=0, output_padding=0),
            self._build_dec_block(4, 4, 10, stride=3, padding=2, output_padding=0),
            self._build_dec_block(4, 1, 10, stride=2, padding=1, output_padding=0),
        )
        return decoder
        
    def _build_dec_block(self, inplane, outplane, kernel_size, stride, padding, output_padding):
        block = nn.Sequential(
            nn.ConvTranspose1d(inplane, outplane, kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.BatchNorm1d(outplane),
            nn.ReLU()
        )
        return block
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
class model_conv1d_autoencoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64, modality="gsr"):
        super(model_conv1d_autoencoder, self).__init__()
        self.n_features, self.embedding_dim = n_features, embedding_dim;

        kernel_size = 5;
        if modality == "gsr":
            channel_list = [4, 8, 16, 32, 64]
            norm = nn.BatchNorm1d
        elif modality == "ecg":
            channel_list = [8, 16, 32, 64, 128]
            norm = nn.BatchNorm1d
            
        # conv1d require the input size: [N,  n_features, seq_len]
        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(self.n_features, channel_list[0], kernel_size, padding=kernel_size//2)),
            ('bn1', norm(channel_list[0])),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool1d(4)),

            ('conv2', nn.Conv1d(channel_list[0], channel_list[1], kernel_size, padding=kernel_size//2)),
            ('bn2', norm(channel_list[1])),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool1d(4)),

            ('conv3', nn.Conv1d(channel_list[1], channel_list[2], kernel_size, padding=kernel_size//2)),
            ('bn3', norm(channel_list[2])),
            ('relu3', nn.ReLU()),
            ('pool3', nn.MaxPool1d(4)),

            ('conv4', nn.Conv1d(channel_list[2], channel_list[3], kernel_size, padding=kernel_size//2)),
            ('bn4', norm(channel_list[3])),
            ('relu4', nn.ReLU()),
            ('pool4', nn.MaxPool1d(4)),

            ('conv5', nn.Conv1d(channel_list[3], channel_list[4], kernel_size, padding=kernel_size//2)),
            ('bn5', norm(channel_list[4])),
            ('relu5', nn.ReLU()),
            ('pool5', nn.MaxPool1d(3)),
        ]))

        self.avgpool = nn.AdaptiveAvgPool1d(1)  # size to be setted
        self.upsample = nn.Upsample(scale_factor=2)   # size to be setted.

        nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)

        self.decoder = nn.Sequential(OrderedDict([
            ('deconv1', nn.ConvTranspose1d(channel_list[4], channel_list[3], kernel_size, stride=3, padding=kernel_size//2, output_padding=2)),
            ('bn6', norm(channel_list[3])),
            ('relu6', nn.ReLU()),
            #('pool5', nn.Upsample(scale_factor=4)),
            
            ('deconv2', nn.ConvTranspose1d(channel_list[3], channel_list[2], kernel_size, stride=4, padding=kernel_size//2, output_padding=3)),
            ('bn7', norm(channel_list[2])),
            ('relu7', nn.ReLU()),
            #('pool5', nn.Upsample(scale_factor=4)),

            ('deconv3', nn.ConvTranspose1d(channel_list[2], channel_list[1], kernel_size, stride=4, padding=kernel_size//2, output_padding=3)),
            ('bn8', norm(channel_list[1])),
            ('relu8', nn.ReLU()),
            #('pool6', nn.Upsample(scale_factor=4)),

            ('deconv4', nn.ConvTranspose1d(channel_list[1], channel_list[0], kernel_size, stride=4, padding=kernel_size//2, output_padding=3)),
            ('bn9', norm(channel_list[0])),
            ('relu9', nn.ReLU()),
            #('pool7', nn.Upsample(scale_factor=4)),

            ('deconv5', nn.ConvTranspose1d(channel_list[0], self.n_features, kernel_size, stride=4, padding=kernel_size//2, output_padding=3)),
            #('pool8', nn.Upsample(scale_factor=4)),
        ]))

    def forward(self, x):
        # x: [B, n_features, seq_len]
        # x = torch.transpose(x, 1, 2)   # [B, seg_len, n_features] => [B, n_features, seq_len]
        x = self.encoder(x);  # [B, self.embedding_dim]
        # print(x.shape)
        feats = self.avgpool(x)
        #print(feats.shape)
        #x = self.upsample(x)
        #print(x.shape)

        x = self.decoder(x)
        # print(x.shape)
        # x = torch.transpose(x, 1, 2)
        return x, feats;