import os, sys
from pyexpat import features
import os.path as osp
import time
import random
import datetime
import argparse
from scipy import linalg
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid
from torchvision.models.inception import inception_v3
from lib.utils import transf_to_CLIP_input, dummy_context_mgr
from lib.utils import mkdir_p, get_rank
from lib.datasets import prepare_data
import img_data as img_data
from models.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist
from scipy.stats import entropy

def calc_IS(imgs,batch_size,device,splits=1):

    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)
    def get_pred(x):
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
        
    N=len(imgs)
    split_scores = []
    preds = np.zeros((N, 1000))
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    for i in range(0, N, batch_size):
        batch = imgs.type(torch.FloatTensor)
        batch = batch.to(device)
        batch_size_i=imgs.size()[0]
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    #compute the mean kl-div 
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

device='cpu'
incep_score=torch.FloatTensor([0.0]).to(device)
path = './samples/images'
print(path)
img_dataset = img_data.Dataset(path, transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
print(img_dataset.__len__())

dataloader = torch.utils.data.DataLoader(dataset=img_dataset, batch_size=200)
dl_length = dataloader.__len__()
for i, imgs in enumerate(dataloader,0):
    inception_score,_=calc_IS(imgs,200,device)
    incep_score=incep_score+inception_score


IS=incep_score.mean().item()/(dl_length)
print(IS)


