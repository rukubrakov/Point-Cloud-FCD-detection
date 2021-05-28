import os
import sys
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import torch.optim as optim

from sklearn.metrics import precision_score, roc_auc_score


DEVICE = 0

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(DEVICE)


def iou(true, pred):
    return np.sum(true & pred) / np.sum(true | pred)

def dice(true, pred):
    return 2*np.sum(true * pred) / (np.sum(true) + np.sum(pred))

def roc_auc_func(targets, probs):
    if (targets.sum() != len(targets)) and (targets.sum() != 0):
        return roc_auc_score(targets, probs)
    else:
        return -1

def add_noise(inputs, std=0.2):
    noise = torch.randn_like(inputs)*std
    return inputs + noise

def train_step(model, opt, criterion, dataloader, std=None):
    roc_auc_score_list = []
    dice_list = []
    iou_list = []
    loss_list = []
    
    model.train()
    for point_clouds in dataloader:
        inp = point_clouds[0].float().transpose(2, 1).to(device)
        target = point_clouds[1].float().to(device)
        
        #if we use noise for augmentation
        if std is not None:
            inp = add_noise(inp, std)
        
        opt.zero_grad()
        preds = model(inp)
        loss = criterion(preds, target)
        loss.backward()
        opt.step()
        
        loss_list.append(float(loss))
        probs = nn.Sigmoid()(preds.detach().cpu()).numpy().reshape(-1)
        pred_labels = (probs > 0.5).astype(int)
        reshaped_targets = target.detach().cpu().numpy().reshape(-1).astype(int)
        
        roc_auc_score_list.append(roc_auc_func(reshaped_targets, probs))
        iou_list.append(iou(reshaped_targets, pred_labels))
        dice_list.append(dice(reshaped_targets, pred_labels))
        
        del inp, target
    
    return np.mean(loss_list), np.mean([el for el in roc_auc_score_list if el != -1]), np.mean(iou_list), np.mean(dice_list)


def crop_val_step(model, opt, criterion, dataloader, std=None):
    """
    validation step in case of cropped dataset
    predictions for all patches in one brain are concatenated 
    """
    pred_list = []
    test_loss = []
    reshaped_targets = []
    
    model.eval()
    with torch.no_grad():
        for point_clouds in dataloader:
            inp = point_clouds[0].float().transpose(2, 1).to(device)
            target = point_clouds[1].float().to(device)

            preds = model(inp)
            loss = criterion(preds, target)

            test_loss.append(float(loss))
            probs = nn.Sigmoid()(preds.detach().cpu()).numpy().reshape(-1)
            pred_list.append(probs)
            
            reshaped_targets.append(target.detach().cpu().numpy().reshape(-1).astype(int))
            
            del inp, target
    
    probs = np.concatenate(pred_list)
    reshaped_targets = np.concatenate(reshaped_targets)
    
    pred_labels = (probs > 0.5).astype(int)
    
    return np.mean(test_loss), roc_auc_score(reshaped_targets, probs), \
           iou(reshaped_targets, pred_labels), dice(reshaped_targets, pred_labels)


def val_step(model, opt, criterion, dataloader, std=None):
    roc_auc_score_list = []
    dice_list = []
    iou_list = []
    loss_list = []
    
    model.eval()
    with torch.no_grad():
        for point_clouds in dataloader:
            inp = point_clouds[0].float().transpose(2, 1).to(device)
            target = point_clouds[1].float().to(device)

            preds = model(inp)
            loss = criterion(preds, target)
            loss_list.append(float(loss))
            
            probs = nn.Sigmoid()(preds.detach().cpu()).numpy().reshape(-1)
            pred_labels = (probs > 0.5).astype(int)
            
            reshaped_targets = target.detach().cpu().numpy().reshape(-1).astype(int)
            
            roc_auc_score_list.append(roc_auc_func(reshaped_targets, probs))
            iou_list.append(iou(reshaped_targets, pred_labels))
            dice_list.append(dice(reshaped_targets, pred_labels))

            del inp, target
            
    return np.mean(loss_list), np.mean([el for el in roc_auc_score_list if el != -1]), np.mean(iou_list), np.mean(dice_list)


#train function
def train_val(model, opt, criterion, train_loader, test_loader, n_epochs, is_train=True, val_step_func=crop_val_step, std=None):
    loss_list = []
    roc_auc_score_list = []
    iou_list = []
    dice_list = []
    
    for epoch in range(n_epochs): 
        if is_train:
            train_loss, train_roc_auc_score, train_iou, train_dice = train_step(model, opt, criterion, train_loader, std)
        
        val_loss, val_roc_auc_score, val_iou, val_dice = val_step_func(model, opt, criterion, test_loader, std)
        
        #clear_output(wait=True)  
        print(f"Epoch = {epoch}, Train loss: {round(train_loss, 3) if is_train else None}, Test loss: {round(val_loss, 3)}")
        print(f"Train ROC-AUC: {round(train_roc_auc_score, 3) if is_train else None}, Test ROC-AUC: {round(val_roc_auc_score, 3)}")
        print(f"Train IoU: {round(train_iou, 3) if is_train else None}, Test IoU: {round(val_iou, 3)}")
        print(f"Train Dice: {round(train_dice, 3) if is_train else None}, Test Dice: {round(val_dice, 3)}")
        
        loss_list.append(val_loss)
        roc_auc_score_list.append(val_roc_auc_score)
        iou_list.append(val_iou)
        dice_list.append(val_dice)
        
    
    return {"loss": np.mean(loss_list), "roc": np.mean([roc for roc in  roc_auc_score_list if roc != -1]),
            "iou": np.mean(iou_list), "dice": np.mean(dice_list)}
