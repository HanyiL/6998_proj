import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from preprocess import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from pycocotools.coco import COCO
import json
from torchtext.data.metrics import bleu_score
import nltk
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction

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

  given_list = []
  pred_list = []
  bleu1 = 0
  bleu2 = 0
  bleu3 = 0
  bleu4 = 0
  print(len(image_ids))
  for i, (image, caption, lengths) in enumerate(data_loader):
    if i % 1000 == 0:
      print(i)
    if i >=40504:
      break
    given_dict = {}
    pred_dict = {}
    image = image.to(device)
    caption = caption.to(device)
    target = pack_padded_sequence(caption, lengths, batch_first=True)[0]
    given_caption = []
    for word_id in target.tolist():
        word = vocab.idx2word[word_id]
        given_caption.append(word)
        if word == '<end>':
            break
    given_caption = given_caption[1:]
    given_caption = given_caption[:-1]
    given_sentence = ' '.join(given_caption)
    given_dict["image_id"] = str(image_ids[i])
    given_dict["caption"] = given_sentence
    given_list.append(given_dict)

    feature = encoder(image)
    sampled_ids = decoder.sample(feature)
    tested_caption = []
    for word_id in sampled_ids.tolist()[0]:
        word = vocab.idx2word[word_id]
        tested_caption.append(word)
        if word == '<end>':
            break
    tested_caption = tested_caption[1:]
    gtested_caption = tested_caption[:-1]
    test_sentence = ' '.join(tested_caption)
    pred_dict["image_id"] = str(image_ids[i])
    pred_dict["caption"] = test_sentence
    pred_list.append(pred_dict)

    #print(nltk.translate.bleu_score.sentence_bleu(given_caption,tested_caption))
    smoothie = SmoothingFunction().method4
    bleu1 += bleu(given_caption,tested_caption, smoothing_function=smoothie,weights=(1, 0, 0, 0))
    bleu2 += bleu(given_caption,tested_caption, smoothing_function=smoothie,weights=(0, 1, 0, 0))
    bleu3 += bleu(given_caption,tested_caption, smoothing_function=smoothie,weights=(0, 0, 1, 0))
    bleu4 += bleu(given_caption,tested_caption, smoothing_function=smoothie,weights=(0, 0, 0, 1))

  if not os.path.exists(args.trainjson_path):
    os.makedirs(args.trainjson_path)
  
  if not os.path.exists(args.valjson_path):
    os.makedirs(args.valjson_path)

  trainjson_dump = json.dumps(given_list)
  torch.save(trainjson_dump, os.path.join(
                    args.trainjson_path, 'givencaptions.json'))
    
  valjson_dump = json.dumps(pred_list)
  torch.save(valjson_dump, os.path.join(
                    args.valjson_path, 'valcaptions_[modelName].json'))
  
  print(bleu1)
  print(bleu2)  
  print(bleu3)  
  print(bleu4)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/content/gdrive/MyDrive/6998_proj/models_[modelName]/' , help='path for saving trained models')
    parser.add_argument('--trainjson_path', type=str, default='/content/gdrive/MyDrive/6998_proj/trainjson/' , help='path for saving validation captions')
    parser.add_argument('--valjson_path', type=str, default='/content/gdrive/MyDrive/6998_proj/valjson_[modelName]/' , help='path for saving validation captions')
    parser.add_argument('--vocab_path', type=str, default='/content/gdrive/MyDrive/6998_proj/data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/content/gdrive/MyDrive/6998_proj/data/val2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='/content/gdrive/MyDrive/6998_proj/data/annotations/captions_val2014.json', help='path for train annotation json file')
    
    parser.add_argument('--encoder_path', type=str, default='models_[modelName]/encoder-5-3000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models_[modelName]/decoder-5-3000.ckpt', help='path for trained decoder')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    print(args)
    main(args)