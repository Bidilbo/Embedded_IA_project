import torch.nn as nn
import torch.nn.functional as F

class ML_Perceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_size, output_size) 
            
    def forward(self, x):
        x = self.flatten(x)
        print(x.shape)
        x = self.fc1(x)  
        x = self.relu1(x)  
        x = self.fc2(x)  
        return x

class ML_Perceptron2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()

        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(input_size, hidden_size1) 
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) 
        self.relu2 = nn.ReLU()  
        self.fc3 = nn.Linear(hidden_size2, output_size) 
            
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)  
        x = self.relu1(x)  
        x = self.fc2(x)
        x = self.relu2(x) 
        x = self.fc3(x)  
        return x