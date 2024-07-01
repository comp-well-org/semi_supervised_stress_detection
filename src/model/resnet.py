from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x;

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResEncoder(nn.Module):
    def __init__(self, layers, inchannel, block=BasicBlock):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(inchannel, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)  # TODO
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        
    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1):
        downsample = None;
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                    nn.Conv1d(self.inplanes, planes*block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(planes*block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride,  downsample=downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))
        
        return  nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x);
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        output = x.view(x.size(0), -1)
        return output
        
class model_ResNet_SMILE(nn.Module):
    def __init__(self, layers, inchannel, block=BasicBlock, num_classes=2, dropout_rate=0.5, is_training=True):
        super(model_ResNet_SMILE, self).__init__()
        self.ecg_encoder = ResEncoder(layers, inchannel=inchannel, block=block)
        self.gsr_encoder = ResEncoder(layers, inchannel=inchannel, block=block)
                
        self.fc = nn.Linear(1024, num_classes)   # the value is undecided yet.
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, ecg, gsr):
        ecg_embedding = self.ecg_encoder(ecg)
        gsr_embedding = self.gsr_encoder(gsr)
        embedding = torch.cat([ecg_embedding, gsr_embedding], dim=1)
        output = self.fc(embedding)
        return output