import torch
import torch.nn as nn
import configs
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from itertools import cycle

def batch2device(batch, device):
    return (tensor.to(device) for tensor in batch)

def train(model, trainloader, testloader, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs.LR_decay_step, gamma=0.1)
    tb_writer = SummaryWriter(log_dir = configs.save_path, comment=configs.tb_comments)
    criterion = nn.BCEWithLogitsLoss()
    for e in range(configs.epochs):
        model.train()
        epoch_loss_train = []
        epoch_gt = []
        epoch_pred = []
        for batch in tqdm(trainloader):
            batch = batch2device(batch, device)
            ecg, gsr, label = batch
            logits = model(ecg, gsr)
            
            optimizer.zero_grad()
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss_train.append(loss.item())
            epoch_pred.append(logits.detach().cpu())
            epoch_gt.append(label.detach().cpu())
        lr_scheduler.step()
        all_pred = np.vstack(epoch_pred)
        all_true = np.vstack(epoch_gt)
        # convert from one-hot coding to binary label.
        all_pred_binary = np.argmax(all_pred, axis=1)
        all_true_binary = np.argmax(all_true, axis=1)
        print("                         Training:")
        print("Loss: %.4f" %(np.mean(np.array(epoch_loss_train))))
        ACC = accuracy_score(all_true_binary, all_pred_binary)
        F1 = f1_score(all_true_binary, all_pred_binary, average="macro")
        print("Accuracy: %.4f " %(ACC), "F1: %.4f " %(F1))
        print(confusion_matrix(all_true_binary, all_pred_binary))
        tb_writer.add_scalar('Loss/Train', np.mean(np.array(epoch_loss_train)), e)
        tb_writer.add_scalar('Acc/Train', ACC, e)
        tb_writer.add_scalar('F1/Train', F1, e)
        
        
        epoch_loss_test = []
        epoch_gt = []
        epoch_pred = []
        model.eval()
        for batch in tqdm(testloader):
            batch = batch2device(batch, device)
            ecg, gsr, label = batch
            logits = model(ecg, gsr)
            
            loss = criterion(logits, label)
            epoch_loss_test.append(loss.item())
            epoch_pred.append(logits.detach().cpu())
            epoch_gt.append(label.detach().cpu())
        all_pred = np.vstack(epoch_pred)
        all_true = np.vstack(epoch_gt)
        
        np.save(configs.save_path + "epoch_{}_pred.npy".format(e), all_pred)
        np.save(configs.save_path + "epoch_{}_true.npy".format(e), all_true)
        
        # convert from one-hot coding to binary label.
        all_pred_binary = np.argmax(all_pred, axis=1)
        all_true_binary = np.argmax(all_true, axis=1)
        print("                         Val:")
        print("Loss: %.4f" %(np.mean(np.array(epoch_loss_test))))
        ACC = accuracy_score(all_true_binary, all_pred_binary)
        F1 = f1_score(all_true_binary, all_pred_binary, average="macro")
        print("Accuracy: %.4f " %(ACC), "F1: %.4f " %(F1))
        print(confusion_matrix(all_true_binary, all_pred_binary))
        tb_writer.add_scalar('Loss/Test', np.mean(np.array(epoch_loss_test)), e)
        tb_writer.add_scalar('Acc/Test', ACC, e)
        tb_writer.add_scalar('F1/Test', F1, e)
        
def train_semi(model, trainloader, unlabeledloader, testloader, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs.LR_decay_step, gamma=0.1)
    tb_writer = SummaryWriter(log_dir = configs.save_path_log, comment=configs.tb_comments)
    criterion = nn.BCEWithLogitsLoss()
    for e in range(configs.epochs):
        model.train()
        epoch_loss_train = []
        epoch_gt = []
        epoch_pred = []
        for batch in tqdm(trainloader):
            batch_unlabeled = next(iter(unlabeledloader))
            batch = batch2device(batch, device)
            batch_unlabeled = batch2device(batch_unlabeled, device)
            ecg, gsr, label = batch
            ecg_1u, ecg_2u, gsr_1u, gsr_2u = batch_unlabeled
            logits = model(ecg, gsr)
            
            logits_1u = model(ecg_1u, gsr_1u)
            logits_2u = model(ecg_2u, gsr_2u)
            
            optimizer.zero_grad()
            
            targets_u = torch.softmax(logits_1u.detach()/0.4, dim=-1)
            max_probs, _ = torch.max(targets_u, dim=-1)
            mask = max_probs.ge(0.8).float()

            loss_unlabel = (-(targets_u * torch.log_softmax(logits_2u, dim=-1)).sum(dim=-1) * mask).mean()
            loss_supervised = criterion(logits, label)
            loss = 0.5 * loss_unlabel + loss_supervised
            loss.backward()
            optimizer.step()
            
            epoch_loss_train.append(loss.item())
            epoch_pred.append(logits.detach().cpu())
            epoch_gt.append(label.detach().cpu())
        lr_scheduler.step()
        all_pred = np.vstack(epoch_pred)
        all_true = np.vstack(epoch_gt)
        # convert from one-hot coding to binary label.
        all_pred_binary = np.argmax(all_pred, axis=1)
        all_true_binary = np.argmax(all_true, axis=1)
        print("                         Training:")
        print("Loss: %.4f" %(np.mean(np.array(epoch_loss_train))))
        ACC = accuracy_score(all_true_binary, all_pred_binary)
        F1 = f1_score(all_true_binary, all_pred_binary, average="macro")
        print("Accuracy: %.4f " %(ACC), "F1: %.4f " %(F1))
        print(confusion_matrix(all_true_binary, all_pred_binary))
        tb_writer.add_scalar('Loss/Train', np.mean(np.array(epoch_loss_train)), e)
        tb_writer.add_scalar('Acc/Train', ACC, e)
        tb_writer.add_scalar('F1/Train', F1, e)
        
        
        epoch_loss_test = []
        epoch_gt = []
        epoch_pred = []
        model.eval()
        for batch in tqdm(testloader):
            batch = batch2device(batch, device)
            ecg, gsr, label = batch
            logits = model(ecg, gsr)
            
            loss = criterion(logits, label)
            epoch_loss_test.append(loss.item())
            epoch_pred.append(logits.detach().cpu())
            epoch_gt.append(label.detach().cpu())
        all_pred = np.vstack(epoch_pred)
        all_true = np.vstack(epoch_gt)
        # convert from one-hot coding to binary label.
        all_pred_binary = np.argmax(all_pred, axis=1)
        all_true_binary = np.argmax(all_true, axis=1)
        print("                         Val:")
        print("Loss: %.4f" %(np.mean(np.array(epoch_loss_test))))
        ACC = accuracy_score(all_true_binary, all_pred_binary)
        F1 = f1_score(all_true_binary, all_pred_binary, average="macro")
        print("Accuracy: %.4f " %(ACC), "F1: %.4f " %(F1))
        print(confusion_matrix(all_true_binary, all_pred_binary))
        tb_writer.add_scalar('Loss/Test', np.mean(np.array(epoch_loss_test)), e)
        tb_writer.add_scalar('Acc/Test', ACC, e)
        tb_writer.add_scalar('F1/Test', F1, e)