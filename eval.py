import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from pycocotools.coco import COCO
import json

def main(args):

# Device configuration
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  transform = transforms.Compose([
          transforms.ToTensor(), 
          transforms.Normalize((0.485, 0.456, 0.406), 
                              (0.229, 0.224, 0.225))])

  with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

  coco = COCO(args.caption_path)
  image_ids = coco.getImgIds()
  # Create the data loader
  data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
  
  # Build models
  encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
  decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
  encoder = encoder.to(device)
  decoder = decoder.to(device)

  # Load the trained model parameters
  encoder.load_state_dict(torch.load(args.encoder_path))
  decoder.load_state_dict(torch.load(args.decoder_path))

  pred_dict = {}
  print(len(image_ids))
  for i, (image, caption, lengths) in enumerate(data_loader):
    print(i)
    if i >=40504:
      break
    image = image.to(device)
    feature = encoder(image)
    sampled_ids = decoder.sample(feature)
    tested_caption = []
    for word_id in sampled_ids.tolist()[0]:
        word = vocab.idx2word[word_id]
        tested_caption.append(word)
        if word == '<end>':
            break
    test_sentence = ' '.join(tested_caption)
    pred_dict[str(image_ids[i])] = test_sentence
  
  if not os.path.exists(args.valjson_path):
    os.makedirs(args.valjson_path)
    
  valjson_dump = json.dumps(pred_dict)
  torch.save(valjson_dump, os.path.join(
                    args.valjson_path, 'valcaptions_[modelname].json'))
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/content/gdrive/MyDrive/6998_proj/[modelname]/' , help='path for saving trained models')
    parser.add_argument('--trainjson_path', type=str, default='/content/gdrive/MyDrive/6998_proj/trainjson/' , help='path for saving validation captions')
    parser.add_argument('--valjson_path', type=str, default='/content/gdrive/MyDrive/6998_proj/valjson_[modelname]/' , help='path for saving validation captions')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='/content/gdrive/MyDrive/6998_proj/data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/content/gdrive/MyDrive/6998_proj/data/val2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='/content/gdrive/MyDrive/6998_proj/data/annotations/captions_val2014.json', help='path for train annotation json file')
    
    parser.add_argument('--encoder_path', type=str, default='models_[modelname]/encoder-5-3000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models_[modelname]/decoder-5-3000.ckpt', help='path for trained decoder')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    print(args)
    main(args)