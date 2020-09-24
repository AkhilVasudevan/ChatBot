import json
from nltk_utilities import *
import numpy as np

from ChatBot import *
from model import *

from torch.utils.data import Dataset, DataLoader

with open('./training_data/train.json','r') as f:
    train_datas=json.load(f)

all_words=[]
tags=[]
xy=[]

# loop through each sentence in our intents patterns
for train_data in train_datas['train']:
    tag = train_data['tag']
    # add to tag list
    tags.append(tag)
    for pattern in train_data['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair - TAG WITH THE TOKENIZED WORD
        xy.append((w, tag))

#remove punctuations & stemming
ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words]

#remove duplicates
all_words=sorted(set(all_words))
tags=sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset=ChatDataset()
train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=NeuralNet(input_size, hidden_size, output_size).to(device)

#Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

#Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words=words.to(device)
        labels=labels.to(dtype=torch.long).to(device)
        
        #Forward pass
        outputs=model(words)
        loss=criterion(outputs, labels)
        
        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#storing trained data
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}
FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')
