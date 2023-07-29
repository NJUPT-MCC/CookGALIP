import torch
import json


from PIL import Image
import clip
import os.path as osp
import os, sys
import torchvision.utils as vutils

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from lib.utils import mkdir_p,get_time_stamp, load_model_weights
from models.CookGALIP import NetG, CLIP_TXT_ENCODER, NetF
from lib.datasets import encode_tokens


time_stamp = get_time_stamp()
device = 'cuda:0' # 'cpu' # 'cuda:0'
CLIP_text = "ViT-B/32"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.eval()

text_encoder = CLIP_TXT_ENCODER(clip_model).to(device)
netG = NetG(64, 100, 512, 256, 3, False, clip_model).to(device)
netF = NetF(1025, 512, device, 32, mixed_precision=False).to(device)
path = './code/saved_models/recipe.pth'
checkpoint = torch.load(path, map_location=torch.device('cpu'))
netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus=False)
netF = load_model_weights(netF, checkpoint['model']['IFBlock'], multi_gpus=False)
batch_size = 32
noise = torch.randn((batch_size, 100)).to(device)

mkdir_p('./samples/images')
with open('captions.txt','r',encoding='utf-8') as f:
        captions=f.read().splitlines()

# generate from text
with torch.no_grad():
    for i in range(len(captions)):
        caption = captions[i]
        caption = caption.split(';')
        print(caption)
        title=caption[0]
        ingredients = caption[1]
        instructions = caption[2].split('.')
        cap = title+ingredients
        tokenized_text = clip.tokenize([cap],truncate=True).to(device)
        sent_emb, word_emb = text_encoder(tokenized_text)

        instruc_tokens = []
        for instruction in instructions:
            instruc_tokens.append(clip.tokenize(instruction,truncate=True).to(device))
        
        sent_instruc_tokens_emb = []
        sent_ingre_tokens_emb=[]
        
        for token in instruc_tokens:
            instruc_tokens_emb, _ = encode_tokens(text_encoder,token)
            instruc_tokens_emb=instruc_tokens_emb.repeat(batch_size,1)
            sent_instruc_tokens_emb.append(instruc_tokens_emb)
        
        sent_emb = sent_emb.repeat(batch_size,1)
        sent_emb = netF(sent_emb, sent_instruc_tokens_emb, sent_ingre_tokens_emb)
        fake_imgs = netG(noise, sent_emb, sent_emb, eval=True).float()
        name = f'{i}'
        vutils.save_image(fake_imgs.data, './samples/images/%s.jpg'%(name), nrow=8, value_range=(-1, 1), normalize=True)
