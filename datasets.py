import os
import os.path as osp
import sys
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import clip as clip
ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)


def get_fix_data(train_dl, test_dl, text_encoder, IFBlock, args):
    fixed_image_train, _, _, fixed_sent_train, fixed_word_train, fixed_key_train= get_one_batch_data(train_dl, text_encoder,  IFBlock, args)
    fixed_image_test, _, _, fixed_sent_test, fixed_word_test, fixed_key_test= get_one_batch_data(test_dl, text_encoder,  IFBlock, args)
    fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
    fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)
    fixed_word = torch.cat((fixed_word_train, fixed_word_test), dim=0)
    fixed_noise = torch.randn(fixed_image.size(0), args.z_dim).to(args.device)
    return fixed_image, fixed_sent, fixed_word, fixed_noise


def get_one_batch_data(dataloader, text_encoder, IFBlock, args):
    data = next(iter(dataloader))
    imgs, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, IFBlock, args.device)
    return imgs, captions, CLIP_tokens, sent_emb, words_embs, keys


def prepare_data(data, text_encoder, IFBlock, device):
    imgs, captions, CLIP_tokens, keys, ingre_tokens, instruc_tokens = data
    imgs, CLIP_tokens = imgs.to(device), CLIP_tokens.to(device)
    sent_emb, words_embs = encode_tokens(text_encoder, CLIP_tokens)

    #sent_ingre_tokens_emb, words_ingre_tokens_embs = encode_tokens(text_encoder,ingre_tokens)
    sent_instruc_tokens_emb=[]
    sent_ingre_tokens_emb = []
    for token in instruc_tokens:
        token=token.to(device)
        instruc_tokens_emb, _ = encode_tokens(text_encoder,token)
        sent_instruc_tokens_emb.append(instruc_tokens_emb)
    for token in ingre_tokens:
        token=token.to(device)
        ingre_tokens_emb, _ = encode_tokens(text_encoder,token)
        sent_ingre_tokens_emb.append(ingre_tokens_emb)
    
    #fus_emb=IFBlock(sent_emb, sent_instruc_tokens_emb, sent_ingre_tokens_emb)
    fus_emb=sent_emb
    return imgs, captions, CLIP_tokens, fus_emb, sent_emb, keys


def encode_tokens(text_encoder, caption):
    # encode text
    with torch.no_grad():
        sent_emb,words_embs = text_encoder(caption)
        sent_emb,words_embs = sent_emb.detach(), words_embs.detach()
    return sent_emb, words_embs 

def get_imgs(img_path, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)
    return img

def get_vireo(title,ingredients):
    ingre_tokens = []
    instruc_tokens=[]
    for ingredient in ingredients:
        ingre_tokens.append(clip.tokenize(ingredient,truncate=True)[0])
    tmp=torch.as_tensor(torch.zeros([77]),dtype=torch.long)    
    if len(ingre_tokens)>31:
        ingre_tokens=ingre_tokens[:31]
    else:
        for i in range(31-len(ingre_tokens)):
            ingre_tokens.append(tmp)
    ingredients=",".join(ingredients)
    caption = title + ";" +ingredients
    tokens=clip.tokenize(caption,truncate=True)
    return caption, tokens[0], ingre_tokens, instruc_tokens

def get_recipe(title, ingredients ,instructions):

    ingre_tokens = [ ]
    for ingredient in ingredients:
        ingre_tokens.append(clip.tokenize(ingredient,truncate=True)[0])

    instruc_tokens = []
    for instruction in instructions:
        instruc_tokens.append(clip.tokenize(instruction,truncate=True)[0])

    caption = []
    ingredients=",".join(ingredients)
    #caption = title
    #title_tokens = clip.tokenize(title,truncate=True)
    #ingredient_tokens=clip.tokenize(ingredients,truncate=True)
    #tokens = torch.concat((title_tokens, ingredient_tokens))
    caption = title +";" +ingredients
    tokens=clip.tokenize(caption,truncate=True)
    #if len(instruc_tokens)>40 or len(ingre_tokens)>25:
    #    print('len_instruc:',len(instruc_tokens),'len_ingre:',len(ingre_tokens))
    tmp=torch.as_tensor(torch.zeros([77]),dtype=torch.long)
    
    if len(ingre_tokens)>31:
        ingre_tokens=ingre_tokens[:31]
    else:
        for i in range(31-len(ingre_tokens)):
            ingre_tokens.append(tmp)
        
    if len(instruc_tokens)>50:
        instruc_tokens=instruc_tokens[:50]
    else:
        for i in range(50-len(instruc_tokens)):
            instruc_tokens.append(tmp)
    
    #print(len(ingre_tokens),len(instruc_tokens))
    
    return caption, tokens[0], ingre_tokens, instruc_tokens


################################################################
#                    Dataset
################################################################
class TextImgDataset(data.Dataset):
    def __init__(self, split, transform=None, args=None):
        self.transform = transform
        self.clip4text = args.clip4text
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.split=split
        
        self.bbox = None
        self.split_dir = os.path.join(self.data_dir, split)
        self.files = self.get_files(self.data_dir,split)
        self.number_example = len(self.files)

    
    def get_files(self,data_dir,split):
        filepath='%s/%s.json'% (data_dir,split)
        if os.path.isfile(filepath):
            with open(filepath,'r',encoding='utf-8') as f:
                contents=json.load(f)
            print('Load contents from: %s (%d)' % (filepath, len(contents)))
        else:
            contents=[]
        return contents


    def __getitem__(self, index):
        #
        key = self.files[index]
        data_dir = self.data_dir
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        
        if self.dataset_name.lower().find('recipe') != -1:
            if self.split=='train':
                title= key['title']
                ingredients = key['ingredients']
                instructions = key['instructions']
                imgs=np.random.choice(key['images'], size=1)
                img_name = '%s/images/train/%s' % (data_dir, imgs[0])

            
            elif self.split=='test':
                title= key['title']
                ingredients = key['ingredients']
                instructions = key['instructions']
                imgs=np.random.choice(key['images'], size=1)
                img_name = '%s/images/test/%s' % (data_dir, imgs[0])
            else:
                title= key['title']
                ingredients = key['ingredients']
                instructions = key['instructions']
                imgs=np.random.choice(key['images'], size=1)
                img_name = '%s/images/val/%s' % (data_dir, imgs[0])

        else:
            if self.split=='train':
                title= key['title']
                ingredients = key['ingredients']
                instructions= []
                imgs=np.random.choice(key['images'], size=1)
                img_name = '%s/images%s' % (data_dir, imgs[0])

            elif self.split=='test':
                title= key['title']
                ingredients = key['ingredients']
                instructions= []
                imgs=np.random.choice(key['images'], size=1)
                img_name = '%s/images%s' % (data_dir, imgs[0])

            else:
                title= key['title']
                ingredients = key['ingredients']
                instructions= []
                imgs=np.random.choice(key['images'], size=1)
                img_name = '%s/images%s' % (data_dir, imgs[0])
        #
        imgs = get_imgs(img_name, bbox, self.transform, normalize=self.norm)
        #caps,tokens = get_caption(text_name,self.clip4text)
        if self.dataset_name.lower().find('recipe') != -1:
            caps, tokens, ingre_tokens, instruc_tokens = get_recipe(title,ingredients, instructions)
        else:
            caps, tokens, ingre_tokens, instruc_tokens = get_vireo(title,ingredients)
        key=key['id']
        return imgs, caps,tokens, key, ingre_tokens, instruc_tokens

    def __len__(self):
        return len(self.files)