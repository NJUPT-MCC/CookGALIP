import os, sys
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import clip
import importlib
from lib.utils import choose_model
from lib.datasets import TextImgDataset as Dataset
from models.CookGALIP import NetF

###########   preparation   ############
def load_clip(clip_info, device):
    import clip as clip
    model = clip.load(clip_info['type'], device=device)[0]
    return model


def prepare_models(args):
    device = args.device
    local_rank = args.local_rank
    multi_gpus = args.multi_gpus
    CLIP4trn = load_clip(args.clip4trn, device).eval()
    CLIP4evl = load_clip(args.clip4evl, device).eval()
    NetG,NetD,NetC,CLIP_IMG_ENCODER,CLIP_TXT_ENCODER = choose_model(args.model)
   
    # image encoder
    CLIP_img_enc = CLIP_IMG_ENCODER(CLIP4trn).to(device)
    for p in CLIP_img_enc.parameters():
        p.requires_grad = False
    CLIP_img_enc.eval()
    # text encoder
    CLIP_txt_enc = CLIP_TXT_ENCODER(CLIP4trn).to(device)
    for p in CLIP_txt_enc.parameters():
        p.requires_grad = False
    CLIP_txt_enc.eval()
    # GAN models
    netG = NetG(args.nf, args.z_dim, args.cond_dim, args.imsize, args.ch_size, args.mixed_precision, CLIP4trn).to(device)
    netD = NetD(args.nf, args.imsize, args.ch_size, args.mixed_precision).to(device)
    netC = NetC(args.nf, args.cond_dim, args.mixed_precision).to(device)
    if (args.multi_gpus) and (args.train):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG = torch.nn.parallel.DistributedDataParallel(netG, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
        netD = torch.nn.parallel.DistributedDataParallel(netD, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
        netC = torch.nn.parallel.DistributedDataParallel(netC, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
     
    #ingredient fusion block
    IFBlock = NetF(args.cond_dim*2+1, args.cond_dim, device, args.batch_size, args.mixed_precision).to(device)
    if (args.multi_gpus) and (args.train):
        IFBlock = torch.nn.parallel.DistributedDataParallel(IFBlock, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
    return CLIP4trn, CLIP4evl, CLIP_img_enc, CLIP_txt_enc, netG, netD, netC, IFBlock


def prepare_dataset(args, split, transform):
    if args.ch_size!=3:
        imsize = 256
    else:
        imsize = args.imsize
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip(),
            ])
    
    dataset = Dataset(split=split, transform=image_transform, args=args)
    return dataset


def prepare_datasets(args, transform):
    # train dataset
    train_dataset = prepare_dataset(args, split='train', transform=transform)
    # test dataset
    val_dataset = prepare_dataset(args, split='test', transform=transform)
    return train_dataset, val_dataset


def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    #imgs = torch.tensor(batch[0], dtype=torch.int32)
    imgs=batch[0]
    caps = batch[1]
    tokens=batch[2]
    key = batch[3]
    del batch
    return imgs,caps,tokens, key

def prepare_dataloaders(args, transform=None):
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, valid_dataset = prepare_datasets(args, transform)
    # train dataloader
    if args.multi_gpus==True:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, sampler=train_sampler)
    else:
        train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, shuffle='False')
    # valid dataloader
    if args.multi_gpus==True:
        valid_sampler = DistributedSampler(valid_dataset)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=2*batch_size, drop_last=True,
            num_workers=num_workers, sampler=valid_sampler)
    else:
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=2*batch_size, drop_last=True,
            num_workers=num_workers, shuffle='False')
    return train_dataloader, valid_dataloader, \
            train_dataset, valid_dataset, train_sampler

