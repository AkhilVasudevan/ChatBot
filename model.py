import torch
import torch.nn as nn

#feeddorward neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        #layer1 
        self.l1 = nn.Linear(input_size, hidden_size) 
        #layer2
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        #layer3
        self.l3 = nn.Linear(hidden_size, num_classes)
        #activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out