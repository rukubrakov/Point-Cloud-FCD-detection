import os
import sys
import glob
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
from torchvision import transforms
import datasets.data_utils as d_utils

import imp

import utils.data_processor as data_processor
imp.reload(data_processor)
from utils.data_processor import *

import utils.visualization_tools as visualization_tools
imp.reload(visualization_tools)
from utils.visualization_tools import *


def make_coin_flip():
    return np.random.choice(2, 1).item()

def give_random_point(sagital_shape, coronal_shape, axial_shape, crope_size):
    return (random.randint(0, sagital_shape-crope_size),\
            random.randint(0, coronal_shape-crope_size),\
            random.randint(0, axial_shape-crope_size)
           )

def give_random_point_inside_fcd(mask):
    
    for_sagital = mask.max((1,2))
    sagital_min = for_sagital.argmax()
    sagital_max = len(for_sagital) - for_sagital[::-1].argmax()
    
    for_coronal = mask.max((0,2))
    coronal_min = for_coronal.argmax()
    coronal_max = len(for_coronal) - for_coronal[::-1].argmax()
    
    for_axial = mask.max((0,1))
    axial_min = for_axial.argmax()
    axial_max = len(for_axial) - for_axial[::-1].argmax()
    
    return (random.randint(sagital_min, sagital_max),\
            random.randint(coronal_min, coronal_max),\
            random.randint(axial_min, axial_max)
           )


def get_random_crope(brain, mask, crope_size = 64, brain_prop_in_crop = 0.5):  
    sagital_shape, coronal_shape, axial_shape = brain.shape
    coin_flip = make_coin_flip()
    if coin_flip == 1:
        min_point = give_random_point(sagital_shape, coronal_shape, axial_shape, crope_size)
        is_break = 0
        while (mask[min_point] == 1.) or (is_break == 0):
            min_point = give_random_point(sagital_shape, coronal_shape, axial_shape, crope_size)
            brain_crop = brain[min_point[0]:min_point[0]+crope_size,\
                 min_point[1]:min_point[1]+crope_size,\
                 min_point[2]:min_point[2]+crope_size]
            if (brain_crop.sum() != 0) and (brain_crop.shape[0]==brain_crop.shape[1]) and (brain_crop.shape[1] == brain_crop.shape[2]) and (brain_crop.shape[1] == crope_size) and ((brain_crop>0).sum() >= (crope_size**3 * brain_prop_in_crop)):
                is_break = 1
            else:
                is_break = 0
        return brain_crop,\
                mask[min_point[0]:min_point[0]+crope_size,\
                     min_point[1]:min_point[1]+crope_size,\
                     min_point[2]:min_point[2]+crope_size]

    elif coin_flip == 0:
        is_break = 0
        min_point_inside_fcd = give_random_point_inside_fcd(mask)
        while (mask[min_point_inside_fcd] == 0.) or (is_break == 0):
            min_point_inside_fcd = give_random_point_inside_fcd(mask)
            brain_crop = brain[min_point_inside_fcd[0]-crope_size//2:min_point_inside_fcd[0]+crope_size//2,\
                     min_point_inside_fcd[1]-crope_size//2:min_point_inside_fcd[1]+crope_size//2,\
                     min_point_inside_fcd[2]-crope_size//2:min_point_inside_fcd[2]+crope_size//2]
            if (brain_crop.sum() != 0) and (brain_crop.shape[0]==brain_crop.shape[1]) and (brain_crop.shape[1] == brain_crop.shape[2]) and (brain_crop.shape[1] == crope_size) and ((brain_crop>0).sum() >= (crope_size**3 * brain_prop_in_crop)):
                is_break = 1
            else:
                is_break = 0
        return brain_crop,\
                mask[min_point_inside_fcd[0]-crope_size//2:min_point_inside_fcd[0]+crope_size//2,\
                     min_point_inside_fcd[1]-crope_size//2:min_point_inside_fcd[1]+crope_size//2,\
                     min_point_inside_fcd[2]-crope_size//2:min_point_inside_fcd[2]+crope_size//2]
    
def get_certain_crop(object_to_crop, sagital_iter, coronal_iter, axial_iter, crope_size):
    sagital_shape, coronal_shape, axial_shape = object_to_crop.shape

    if sagital_iter == sagital_shape//crope_size:
        return object_to_crop[sagital_iter*crope_size:,\
                          coronal_iter*crope_size:(coronal_iter+1)*crope_size,\
                          axial_iter*crope_size:(axial_iter+1)*crope_size]
    if coronal_iter == coronal_shape//crope_size:
        return object_to_crop[sagital_iter*crope_size:(sagital_iter+1)*crope_size,\
                          coronal_iter*crope_size:,\
                          axial_iter*crope_size:(axial_iter+1)*crope_size]
    
    if axial_iter == axial_shape//crope_size:
        return object_to_crop[sagital_iter*crope_size:(sagital_iter+1)*crope_size,\
                          coronal_iter*crope_size:(coronal_iter+1)*crope_size,\
                          axial_iter*crope_size:]
    return object_to_crop[sagital_iter*crope_size:(sagital_iter+1)*crope_size,\
                      coronal_iter*crope_size:(coronal_iter+1)*crope_size,\
                      axial_iter*crope_size:(axial_iter+1)*crope_size]

def get_inference_cropes(brain, mask, crope_size = 64,is_return_center = False):
    
    sagital_shape, coronal_shape, axial_shape = brain.shape
    brain_crops = []
    mask_crops = []
    center_coords = []
    for sagital_iter in range(sagital_shape//crope_size+1):
        for coronal_iter in range(coronal_shape//crope_size+1):
            for axial_iter in range(axial_shape//crope_size+1):
                croped_brain = get_certain_crop(brain, sagital_iter, coronal_iter, axial_iter, crope_size)
                if croped_brain.sum() != 0.:
                    brain_crops.append(croped_brain)
                    mask_crops.append(get_certain_crop(mask, sagital_iter, coronal_iter, axial_iter, crope_size))
                    center_coords.append((sagital_iter*crope_size,coronal_iter*crope_size,axial_iter*crope_size))
    if is_return_center:
        return brain_crops, mask_crops, center_coords
    else:
        return brain_crops, mask_crops


def brain_and_mask_to_point_cloud_and_labels(brain, mask,crope_size = 64, is_crop = True):
    if is_crop:
        brain,mask = get_random_crope(brain, mask, crope_size = crope_size)
    size = brain.shape
    grid_x, grid_y, grid_z = torch.meshgrid((torch.tensor(range(size[0])) - (size[0]/2)) / (size[0]/2),\
                                            (torch.tensor(range(size[1])) - (size[1]/2)) / (size[1]/2),\
                                            (torch.tensor(range(size[2])) - (size[2]/2)) / (size[2]/2))

    point_cloud = torch.cat((grid_x.unsqueeze(-1).float(), 
                     grid_y.unsqueeze(-1).float(),
                     grid_z.unsqueeze(-1).float(), 
                     ((torch.tensor(brain)-300)/300).float().unsqueeze(-1).float()), -1)
    point_cloud_fcd = point_cloud[mask == 1, :]
    pc_brain_without_fcd = point_cloud[(mask==0)*(brain != 0),:]
    
    return torch.cat([point_cloud_fcd,pc_brain_without_fcd]),\
        np.array([1] * point_cloud_fcd.shape[0] + [0] * pc_brain_without_fcd.shape[0])


class BrainDataSegCrop():
    def __init__(self, barin_paths='../pytorch/croped_new_dataset/fcd_brains',
                 mask_paths='../pytorch/croped_new_dataset/masks', task = 'train',crope_size = 64, test_brain = 0,
                 num_points = 2048,transforms = None,is_return_center = False):
        self.transforms = transforms
        self.is_return_center = is_return_center
        self.task = task
        self.crope_size = crope_size
        self.num_points = num_points
        if task == 'train':
            self.brains = sorted([x for x in glob.glob(f'{barin_paths}/*') if ('no' not in x) and ('1.nii.gz' in x) and (int(x.split('/')[-1].split('_')[-1][:2])!=test_brain)])
            self.masks = [x.replace('fcd_brains/fcd','masks/mask_fcd').replace('fcd_brains','masks') for x in self.brains]
            self.brains_loaded = [load_nii_to_array(brain_path) for brain_path in self.brains]
            self.masks_loaded = [load_nii_to_array(mask_path) for mask_path in self.masks]
        elif task == 'test':
            brain_path = [x for x in glob.glob(f'{barin_paths}/*') if ('no' not in x) and ('1.nii.gz' in x) and (int(x.split('/')[-1].split('_')[-1][:2])==test_brain)][0] 
            mask_path = brain_path.replace('fcd_brains/fcd','masks/mask_fcd').replace('fcd_brains','masks')
            brain = load_nii_to_array(brain_path)
            mask = load_nii_to_array(mask_path)
            tmp = get_inference_cropes(brain, mask, crope_size = crope_size,is_return_center = is_return_center)
            if is_return_center:
                brain_crops, mask_crops, center_coords = tmp
                self.center_coords = center_coords
            else:
                brain_crops, mask_crops = tmp
            points_labels = [brain_and_mask_to_point_cloud_and_labels(brain, mask,crope_size = self.crope_size, is_crop = False) for brain, mask in zip(brain_crops,mask_crops)]
            self.points = [x[0] for x in points_labels]
            self.labels = [x[1] for x in points_labels]
            
    def __getitem__(self, idx):
        
        if self.task == 'train':
            brain = self.brains_loaded[idx]
            mask = self.masks_loaded[idx]
            current_points, current_points_labels = brain_and_mask_to_point_cloud_and_labels(brain, mask,crope_size = self.crope_size)
        elif self.task == 'test':
            current_points, current_points_labels = self.points[idx],self.labels[idx]
            
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
        label = torch.from_numpy(np.array(0)).type(torch.int64)
        current_points_labels = torch.from_numpy(current_points_labels).type(torch.int64)
        if self.is_return_center:
            return current_points, mask, current_points_labels, label, self.center_coords[idx]
        else:
            return current_points, mask, current_points_labels, label
    def __len__(self):
        if self.task == 'train':
            return len(self.brains)
        elif self.task == 'test':
            return len(self.labels)
    
def get_loader_crop(config,
                    num_points = 2048,
                    batch_size = 16,
                    test_brain = 0,
                    crope_size = 64, 
                    barin_paths='../pytorch/croped_new_dataset/fcd_brains',
                    mask_paths='../pytorch/croped_new_dataset/masks'):
                     
    trans_test = None
    
    trans_train = transforms.Compose([
        d_utils.PointcloudRandomRotate(x_range=config.x_angle_range, y_range=config.y_angle_range,
                                       z_range=config.z_angle_range),
        d_utils.PointcloudScaleAndJitter(scale_low=config.scale_low, scale_high=config.scale_high,
                                         std=config.noise_std, clip=config.noise_clip,
                                         augment_symmetries=config.augment_symmetries),
        d_utils.PointcloudTranslate(translate_range = 0.05)
    ])
    train_dataset = BrainDataSegCrop(num_points=num_points,
                                     task = 'train',
                                     crope_size = crope_size,
                                     test_brain = test_brain,
                                     barin_paths = barin_paths,
                                     transforms = trans_train, 
                                     mask_paths = mask_paths
                                )
    test_dataset = BrainDataSegCrop(num_points=num_points,
                                    task = 'test',
                                    crope_size = crope_size,
                                    test_brain = test_brain,
                                    barin_paths = barin_paths,
                                    transforms = trans_test,
                                    mask_paths = mask_paths)
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

    return train_loader, test_loader