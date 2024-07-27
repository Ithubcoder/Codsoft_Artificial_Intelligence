import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained ResNet model
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

# Define the RNN Decoder
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

# Load image and transform
def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# Vocabulary (this should be pre-built and trained based on your dataset)
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

# Initialize vocabulary (In practice, this should be loaded from a pre-trained vocabulary)
vocab = Vocabulary()
vocab.add_word('<pad>')
vocab.add_word('<start>')
vocab.add_word('<end>')
vocab.add_word('<unk>')
# Add more words as necessary...

# Image captioning model
embed_size = 256
hidden_size = 512
num_layers = 1
vocab_size = len(vocab)

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

# Load an example image
image_path = "C:/Users/Mukul/Desktop/6.webp"
image = load_image(image_path, transform)

# Extract features and generate caption
encoder.eval()
decoder.eval()

with torch.no_grad():
    features = encoder(image)
    sampled_ids = []
    inputs = features.unsqueeze(1)
    states = None

    for i in range(20):  # Maximum sampling length
        hiddens, states = decoder.lstm(inputs, states)
        outputs = decoder.linear(hiddens.squeeze(1))
        _, predicted = outputs.max(1)
        sampled_ids.append(predicted.item())
        inputs = decoder.embed(predicted)
        inputs = inputs.unsqueeze(1)

        if vocab.idx2word[predicted.item()] == '<end>':
            break

    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    print(sentence)

# Display the image and caption
image = Image.open(image_path)
plt.imshow(np.asarray(image))
plt.title(sentence)
plt.show()