import os
import json
import time
import shlex
import glob
import torch
import pickle
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import datasets.data_utils as d_utils
import pandas as pd
from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn.metrics import accuracy_score
from utils.util import AverageMeter, shapenetpart_metrics
from models.backbones import ResNet
from models.heads import ClassifierResNet, MultiPartSegHeadResNet, SceneSegHeadResNet
from models.losses import LabelSmoothingCrossEntropyLoss, MultiShapeCrossEntropy, MaskedCrossEntropy
from utils.config import config, update_config
from utils.lr_scheduler import get_scheduler
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
from IPython.display import clear_output
from tqdm import tqdm
import pickle
import warnings
from utils.pytorchtools import EarlyStopping


class BrainDataSeg():
    def __init__(self, data_type = 'train', num_points = 2048, 
                 transforms=None,
                 data_post = '',
                 datafolder = 'BrainData'):
        self.num_points = num_points
        self.transforms = transforms
        if data_type == 'test':
            filename = f'data/{datafolder}/test_data{data_post}.pkl'
        if data_type == 'train':
            filename = f'data/{datafolder}/trainval_data{data_post}.pkl'
        with open(filename, 'rb') as f:
            self.points, self.points_labels, self.labels = pickle.load(f)
        print(f"{filename} loaded successfully")

    def __getitem__(self, idx):
        
        current_points = self.points[idx]
        current_points_labels = self.points_labels[idx]
        cur_num_points = current_points.shape[0]
        if cur_num_points >= self.num_points:
            choice = np.random.choice(cur_num_points, self.num_points)
            current_points = current_points[choice, :]
            current_points_labels = current_points_labels[choice]
            mask = torch.ones(self.num_points).type(torch.int32)
        else:
            padding_num = self.num_points - cur_num_points
            shuffle_choice = np.random.permutation(np.arange(cur_num_points))
            padding_choice = np.random.choice(cur_num_points, padding_num)
            choice = np.hstack([shuffle_choice, padding_choice])
            current_points = current_points[choice, :]
            current_points_labels = current_points_labels[choice]
            mask = torch.cat([torch.ones(cur_num_points), torch.zeros(padding_num)]).type(torch.int32)
        if self.transforms is not None:
            current_points = self.transforms(current_points)
        label = torch.from_numpy(self.labels[idx]).type(torch.int64)
        current_points_labels = torch.from_numpy(current_points_labels).type(torch.int64)

        return current_points, mask, current_points_labels, label

    def __len__(self):
        return len(self.points)
    
    
def get_loader(num_points,batch_size = 16,data_post = '', datafolder = 'BrainData'):
    trans_test = transforms.Compose([d_utils.PointcloudToTensor()])
    trans_train = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRandomRotate(x_range=config.x_angle_range, y_range=config.y_angle_range,
                                       z_range=config.z_angle_range),
        d_utils.PointcloudScaleAndJitter(scale_low=config.scale_low, scale_high=config.scale_high,
                                         std=config.noise_std, clip=config.noise_clip,
                                         augment_symmetries=config.augment_symmetries),
    d_utils.PointcloudTranslate(translate_range = 0.05)])
    train_dataset = BrainDataSeg(num_points=num_points,
                                 data_type = 'train',
                                 transforms = trans_train, 
                                 data_post = data_post,
                                 datafolder = datafolder
                                )
    test_dataset = BrainDataSeg(num_points=num_points,
                                data_type = 'test',
                                transforms = trans_test, 
                                data_post = data_post,
                               datafolder = datafolder)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              drop_last=False)

    return train_loader, test_loader, train_dataset.points_labels

class MultiPartSegmentationModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes, num_parts,
                 input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        super(MultiPartSegmentationModel, self).__init__()
        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Multi-Part Segmentation Model")

        if head == 'resnet_part_seg':
            self.segmentation_head = MultiPartSegHeadResNet(num_classes, width, radius, nsamples, num_parts)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Multi-Part Segmentation Model")
    def forward(self, xyz, mask, features):
        end_points = self.backbone(xyz, mask, features)
        return self.segmentation_head(end_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

def config_seting(cfg = 'cfgs/brain/brain.yaml'):
    update_config(cfg)
    return config

def build_multi_part_segmentation(config, weights = None, type = 'MultiShape',is_kuni = False,
                                  kuni_agg = 'mean',kuni_lam = 0.001,is_sep_loss = False):
    model = MultiPartSegmentationModel(config, config.backbone, config.head, config.num_classes, config.num_parts,
                                       config.input_features_dim,
                                       config.radius, config.sampleDl, config.nsamples, config.npoints,
                                       config.width, config.depth, config.bottleneck_ratio)
    criterion = MultiShapeCrossEntropy(config.num_classes, type = type,weights = weights,
                                       is_kuni = is_kuni,kuni_agg = kuni_agg,
                                       kuni_lam = kuni_lam,is_sep_loss = is_sep_loss)
    return model, criterion

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate  """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i),'tpr' : pd.Series(tpr, index=i),'fpr' : pd.Series(fpr, index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return np.mean(list(roc_t['threshold'])), list(roc_t['tpr'])[0], list(roc_t['fpr'])[0]

def train(epoch, train_loader, model, criterion, optimizer, scheduler, config, is_kuni = False,is_sep_loss = False):
    """ One epoch training """
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    if is_sep_loss:
        loss_meter2 = AverageMeter()
    end = time.time()
    pred_soft_flats = []
    points_labels_flats = []
    for idx, (points, mask, points_labels, shape_labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = points.size(0)
        # forward
        features = points
        features = features.transpose(1, 2).contiguous()

        points = points[:,:,:3].cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        features = features.cuda(non_blocking=True)
        points_labels = points_labels.cuda(non_blocking=True)
        shape_labels = shape_labels.cuda(non_blocking=True)
        pred = model(points, mask, features)
        if is_kuni:
            loss = criterion(pred, points_labels, shape_labels,points)
        else:
            loss = criterion(pred, points_labels, shape_labels)
        
        
        if is_sep_loss:
            loss_sum = loss[0] + loss[1]
        m = torch.nn.Softmax(dim=1)
        pred_soft_flats += list(np.array(m(pred[0])[:,1,:].reshape(-1).detach().cpu()))
        points_labels_flats += list(np.array(points_labels.reshape(-1).detach().cpu()))

        optimizer.zero_grad()
        if is_sep_loss:
            loss_sum.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        # update meters
        if is_sep_loss:
            loss_meter.update(loss[0].item(), bsz)
            loss_meter2.update(loss[1].item(), bsz)
        else:
            loss_meter.update(loss.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()
        del points, mask, points_labels, shape_labels, loss
    opt,tpr,fpr = Find_Optimal_Cutoff(points_labels_flats, pred_soft_flats)
    if is_sep_loss:
        return (loss_meter.avg,loss_meter2.avg), opt, roc_auc_score(points_labels_flats,pred_soft_flats)
    else:
        return loss_meter.avg, opt, roc_auc_score(points_labels_flats,pred_soft_flats)

def validate(epoch, test_loader, model, criterion, config, num_votes=10,
             is_conf = False,is_kuni = False,is_sep_loss = False, is_crop = False):
    """ One epoch validating """
    batch_time = AverageMeter()
    
    losses = AverageMeter()
    if is_sep_loss:
        losses2 = AverageMeter()
    model.eval()
    with torch.no_grad():
        all_logits = []
        all_points_labels = []
        all_shape_labels = []
        all_masks = []
        end = time.time()
        TS = d_utils.BatchPointcloudScaleAndJitter(scale_low=config.scale_low,
                                                   scale_high=config.scale_high,
                                                   std=config.noise_std,
                                                   clip=config.noise_clip)
        pred_soft_flats = []
        points_labels_flats = []
        for idx, (points_orig, mask, points_labels, shape_labels) in enumerate(test_loader):
            vote_logits = None
            vote_points_labels = None
            vote_shape_labels = None
            vote_masks = None
            for v in range(num_votes):
                batch_logits = []
                batch_points_labels = []
                batch_shape_labels = []
                batch_masks = []
                # augment for voting
                if v > 0:
                    points = TS(points_orig)
                else:
                    points = points_orig
                # forward
                features = points
                features = features.transpose(1, 2).contiguous()
                points = points[:,:,:3].cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                features = features.cuda(non_blocking=True)
                points_labels = points_labels.cuda(non_blocking=True)
                shape_labels = shape_labels.cuda(non_blocking=True)

                pred = model(points, mask, features)
                if is_kuni:
                    loss = criterion(pred, points_labels, shape_labels,points)
                else:
                    loss = criterion(pred, points_labels, shape_labels)
                if is_sep_loss:
                    losses.update(loss[0].item(), points.size(0))
                    losses2.update(loss[1].item(), points.size(0))
                else:
                    losses.update(loss.item(), points.size(0))
                
                m = torch.nn.Softmax(dim=1)
                pred_soft_flats += list(np.array(m(pred[0])[:,1,:].reshape(-1).detach().cpu()))
                points_labels_flats += list(np.array(points_labels.reshape(-1).detach().cpu()))
                

                # collect
                bsz = points.shape[0]
                for ib in range(bsz):
                    sl = shape_labels[ib]
                    logits = pred[sl][ib]
                    pl = points_labels[ib]
                    pmk = mask[ib]
                    batch_logits.append(logits.cpu().numpy())
                    batch_points_labels.append(pl.cpu().numpy())
                    batch_shape_labels.append(sl.cpu().numpy())
                    batch_masks.append(pmk.cpu().numpy().astype(np.bool))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if vote_logits is None:
                    vote_logits = batch_logits
                    vote_points_labels = batch_points_labels
                    vote_shape_labels = batch_shape_labels
                    vote_masks = batch_masks
                else:
                    for i in range(len(vote_logits)):
                        vote_logits[i] = vote_logits[i] + (batch_logits[i] - vote_logits[i]) / (v + 1)

            all_logits += vote_logits
            all_points_labels += vote_points_labels
            all_shape_labels += vote_shape_labels
            all_masks += vote_masks                
            del points_orig, mask, points_labels, shape_labels,loss    
        if is_conf:
            acc, shape_ious, msIoU, mIoU, confs,dice_score = shapenetpart_metrics(config.num_classes,
                                                            config.num_parts,
                                                            all_shape_labels,
                                                            all_logits,
                                                            all_points_labels,
                                                            all_masks, is_conf = is_conf)
        else:
            acc, shape_ious, msIoU, mIoU,dice_score = shapenetpart_metrics(config.num_classes,
                                                            config.num_parts,
                                                            all_shape_labels,
                                                            all_logits,
                                                            all_points_labels,
                                                            all_masks, is_conf = is_conf)
    opt, tpr, fpr = Find_Optimal_Cutoff(points_labels_flats, pred_soft_flats)
    if is_crop:
        msIoU = confs[1,1]/(confs[1,1]+confs[1,0]+confs[0,1])
        dice_score = 2 * confs[1,1] / (2 * confs[1,1]+confs[1,0]+confs[0,1])
        msIoU_proba = np.dot(points_labels_flats,pred_soft_flats)/np.sum(pred_soft_flats)
        dice_proba = 2 * np.dot(points_labels_flats,pred_soft_flats) / (np.sum(pred_soft_flats) + np.dot(points_labels_flats, pred_soft_flats))
        ideal_msIoU = np.dot(points_labels_flats,pred_soft_flats>=opt) / (np.sum(pred_soft_flats>=opt)+np.dot(pred_soft_flats<opt,points_labels_flats))
        ideal_dice = 2 * np.dot(points_labels_flats,pred_soft_flats>=opt) / (np.sum(pred_soft_flats>=opt)+np.sum(points_labels_flats))
        d_in = np.dot(points_labels_flats,pred_soft_flats)/np.sum(points_labels_flats)
        d_out = np.dot(np.ones(len(points_labels_flats))-np.array(points_labels_flats),pred_soft_flats) / (len(points_labels_flats) - np.sum(points_labels_flats))
        contrast = (d_in - d_out) /  (d_in + d_out)
        msIoU = (msIoU, msIoU_proba,ideal_msIoU,contrast)
        dice_score = (dice_score, dice_proba,ideal_dice)
    if is_conf:
        if is_sep_loss:
            return (losses.avg,losses2.avg), acc, msIoU, mIoU, confs, roc_auc_score(points_labels_flats,pred_soft_flats),opt,tpr,fpr,dice_score         
        else:
            return losses.avg,acc, msIoU, mIoU, confs, roc_auc_score(points_labels_flats,pred_soft_flats),opt,tpr,fpr,dice_score
    else:
        if is_sep_loss:
            return (losses.avg,losses2.avg),acc, msIoU, mIoU, roc_auc_score(points_labels_flats,pred_soft_flats),opt,fpr,dice_score            
        else:
            return losses.avg,acc, msIoU, mIoU, roc_auc_score(points_labels_flats,pred_soft_flats),opt,fpr,dice_score