import torch
import numpy as np
from ML_Perceptron import ML_Perceptron

input_size = 28*28
hidden_size = 128
output_size = 10
model = ML_Perceptron(input_size, hidden_size, output_size) 
model.load_state_dict(torch.load("modeles/MLP_custom_dataset_pp.pth"))
model.eval()
print(model)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("number of parameters : ", params)

weights = {}
for name, param in model.named_parameters():
    weights[name] = param.detach().cpu().numpy()

for name, param in weights.items():
    file_name = f"{name.replace('.', '_')}.txt"
    np.savetxt(file_name, param.flatten())  
    print(f"Paramètre {name} sauvegardé dans {file_name}")