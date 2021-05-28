import torch
import torch.nn as nn
import torch.nn.functional as F

class KUNi(nn.Module):
    def __init__(self, agg = 'mean', lam = 0.001):
        super(KUNi, self).__init__()
        self.agg = agg
        self.lam = lam

    def forward(self, inputs, coords):
        '''inputs(B,C,N), coords(B,N,3)'''
        coords = coords[:,:,:3] #(B,N,3)
        d = torch.cdist(coords, coords) #(B,N,N)
        inputs = F.softmax(inputs, dim = 1)[:,1,:].squeeze(1) #(B,N) 
        dpp = d * inputs[:,:,None] * inputs[:,:,None].permute(0,2,1) #(B,N,N)
        
        if self.agg == 'mean':
            return torch.mean(dpp) * self.lam
        if self.agg == 'max':
            return torch.mean(torch.max(dpp,dim = [1,2])) * self.lam
        if self.agg == 'sum':
            return torch.mean(torch.sum(dpp,dim = [1,2])) * self.lam    
class DiceBCELoss(nn.Module):
    def __init__(self, weights=None):
        super(DiceBCELoss, self).__init__()
        if weights is not None:
            self.bce = torch.nn.CrossEntropyLoss(torch.tensor(weights).float().cuda())
        else:
            self.bce = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, targets, smooth=1):
        inputs_for_bce =  inputs.permute(0,2,1)
        inputs_for_bce = inputs_for_bce.reshape(-1, inputs_for_bce.shape[-1])
        targets_for_bce = targets.view(-1)
        inputs = F.softmax(inputs, dim = 1)[:,1,:].squeeze() #(B,C,N) - > (B,N)     

        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = self.bce(inputs_for_bce, targets_for_bce)
        #BCE = F.binary_cross_entropy(inputs, targets.float(), reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class MultiShapeCrossEntropy(nn.Module):
    def __init__(self, num_classes, type = 'MultiShape', weights = None, 
                 is_kuni = False, kuni_agg = 'mean', kuni_lam = 0.001,is_sep_loss = False):
        super(MultiShapeCrossEntropy, self).__init__()
        self.type = type
        self.is_sep_loss = is_sep_loss
        self.is_kuni = is_kuni
        if is_kuni:
            self.kuni = KUNi(kuni_agg,kuni_lam)
        self.num_classes = num_classes
        if weights is not None:
            self.bce = torch.nn.CrossEntropyLoss(torch.tensor(weights).float().cuda())
        else:
            self.bce = torch.nn.CrossEntropyLoss()
        self.dice = DiceBCELoss(weights=weights)

    def forward(self, logits_all_shapes, points_labels, shape_labels, coords = None):
        if self.type=='MultiShape':
            batch_size = shape_labels.shape[0]
            losses = 0
            for i in range(batch_size):
                sl = shape_labels[i]
                logits = torch.unsqueeze(logits_all_shapes[sl][i], 0)
                pl = torch.unsqueeze(points_labels[i], 0)
                loss = F.cross_entropy(logits, pl)
                losses += loss
                for isl in range(self.num_classes):
                    if isl == sl:
                        continue
                    losses += 0.0 * logits_all_shapes[isl][i].sum()

            return losses / batch_size
        elif self.type=='BCE':
            logits_all_shapes1 = logits_all_shapes[0].permute(0,2,1)
            logits_all_shapes1 = logits_all_shapes1.reshape(-1,logits_all_shapes1.shape[-1])
            points_labels = points_labels.view(-1)
            if self.is_kuni:
                if self.is_sep_loss:
                    return self.bce(logits_all_shapes1,points_labels), self.kuni(logits_all_shapes[0],coords)
                else:
                    return self.bce(logits_all_shapes1,points_labels) + self.kuni(logits_all_shapes[0],coords)
            else:
                return self.bce(logits_all_shapes1,points_labels)
        elif self.type=='DICE':
            if self.is_kuni:
                if self.is_sep_loss:
                    return self.dice(logits_all_shapes[0], points_labels), self.kuni(logits_all_shapes[0],coords)
                else:
                    return self.dice(logits_all_shapes[0], points_labels) + self.kuni(logits_all_shapes[0],coords)
            else:
                return self.dice(logits_all_shapes[0], points_labels)