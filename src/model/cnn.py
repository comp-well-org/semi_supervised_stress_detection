import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from model_rl.autoencoder import model_conv1d_autoencoder
import configs

class model_conv1d(nn.Module):
    def __init__(self, num_classes=2, embedding_dim=192, modality="gsr", load_pretrained=True):
        super(model_conv1d, self).__init__()

        self.ecg_encoder = model_conv1d_autoencoder(1, modality="ecg").encoder
        self.gsr_encoder = model_conv1d_autoencoder(1, modality="gsr").encoder
        
        if load_pretrained:
            state_dict = torch.load(configs.save_model_path_ecg)
            model_state = self.ecg_encoder.state_dict()
            pretrained_dict = {k[8:]: v for k, v in state_dict.items() if k[8:] in model_state and "encoder" in k}
            print(pretrained_dict.keys())
            model_state.update(pretrained_dict)
            self.ecg_encoder.load_state_dict(model_state)
            
            state_dict = torch.load(configs.save_model_path_gsr)
            model_state = self.gsr_encoder.state_dict()
            pretrained_dict = {k[8:]: v for k, v in state_dict.items() if k[8:] in model_state and "encoder" in k}
            print(pretrained_dict.keys())
            model_state.update(pretrained_dict)
            self.gsr_encoder.load_state_dict(model_state)
        
        self.avgpool_ecg = nn.AdaptiveAvgPool1d(1)
        self.avgpool_gsr = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(192, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, num_classes)

    def forward(self, ecg, gsr):
        ecg_embedding = self.ecg_encoder(ecg)
        ecg_embedding = self.avgpool_ecg(ecg_embedding)
        gsr_embedding = self.gsr_encoder(gsr)
        gsr_embedding = self.avgpool_gsr(gsr_embedding)
        embedding = torch.cat([ecg_embedding.squeeze(-1), gsr_embedding.squeeze(-1)], dim=1)
        embedding = F.dropout(embedding, 0.3)
        output = F.relu(self.fc1(embedding))
        output = F.dropout(output, 0.3)
        output = F.relu(self.fc2(output))
        output = self.out(output)
        return output