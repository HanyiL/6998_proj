import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, CNNModel):
        super(EncoderCNN, self).__init__()
        if 'res' in CNNModel:
          if CNNModel == 'res18':
              cnnNet = models.resnet18(pretrained=True)
          elif CNNModel == 'res34':
              cnnNet = models.resnet34(pretrained=True)
          elif CNNModel == 'res50':
              cnnNet = models.resnet50(pretrained=True)
          elif CNNModel == 'res101':
              cnnNet = models.resnet101(pretrained=True)
          elif CNNModel == 'res152':
              cnnNet = models.resnet152(pretrained=True)
          modules = list(cnnNet.children())[:-1]      # delete the last fc layer.
          self.cnnNet = nn.Sequential(*modules)
          self.linear = nn.Linear(cnnNet.fc.in_features, embed_size)
          self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        else:
          if CNNModel == 'vgg11':
            self.cnnNet = models.vgg11(pretrained=True)
          elif CNNModel == 'vgg13':
            self.cnnNet = models.vgg13(pretrained=True)
          elif CNNModel == 'vgg16':
            self.cnnNet = models.vgg16(pretrained=True)
          elif CNNModel == 'vgg19':
            self.cnnNet = models.vgg19(pretrained=True)
          modules = list(self.cnnNet.classifier.children())[:-1]
          self.cnnNet.classifier = nn.Sequential(*modules)
          self.linear = nn.Linear(in_features=4096, out_features=embed_size)
          self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)


    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.cnnNet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids