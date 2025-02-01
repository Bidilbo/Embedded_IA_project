import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import shutil

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# PRE TRAITEMENT
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertion en niveaux de gris
    transforms.ToTensor(),  # Convertion en tenseur
])

###### CHARGEMENT DU DATASET ######

# Chemin vers le dossier contenant les images pré-traitées  
dataset_path = "/home/docker/Work/data/dataset_preprocessed"
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
#print(f"Classes: {dataset.classes}")

# Arrangement du dataset
train_indices, test_indices = [], []

# Regrouper les indices par classe
# forme : {0 : [0,1,...,9], 1 : [10,11, ...,19], 2 : ..., 9 : [90,91,...,99]}

class_to_indices = {class_idx: [] for class_idx in range(len(dataset.classes))}
for idx, (_, label) in enumerate(dataset):
    class_to_indices[label].append(idx)

# Séparer les indices pour chaque classe

num_test_img_in_each_class = 2
for class_idx, indices in class_to_indices.items():

    random.shuffle(indices)
    train_indices.extend(indices[num_test_img_in_each_class:])
    test_indices.extend(indices[:num_test_img_in_each_class])


# Création des sous-datasets pour l'entraînement et le test
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Obtenir les chemins des images pour le test
test_images = [dataset.samples[idx][0] for idx in test_indices]

#output_dir = "images_inference"
output_dir = "/home/docker/Work/images_inference"

# Supprimer le dossier s'il existe et tout son contenu
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir, exist_ok=True)

# Copier les images de test dans le dossier de sortie
for idx in test_indices:
    img_path = dataset.samples[idx][0] 
    img_name = os.path.basename(img_path) 
    dest_path = os.path.join(output_dir, img_name)

    shutil.copy(img_path, dest_path)

print("Nombre d'images de test :", len(test_indices))
print("\nImages utilisées pour le test sauvegardées dans le dossier", output_dir)

batch_size = 2
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""for X, y in test_dataloader:
    print(f"Forme de X [N, C, H, W]: {X.shape}")
    print(f"Forme de y: {y.shape} {y.dtype}")
    break"""


######## Entrainement ########

from ML_Perceptron import ML_Perceptron, ML_Perceptron2

input_size = 28*28
#hidden_size = 128
hidden_size1 = 256
hidden_size2 = 128
output_size = 10
#model = ML_Perceptron(input_size, hidden_size, output_size).to(device)
model = ML_Perceptron2(input_size, hidden_size1, hidden_size2, output_size).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Erreur de prédiction
        pred = model(X)
        loss = loss_fn(pred, y)

        # Propagation arrière
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Entrainement terminé.")

model_save_dir = "/home/docker/Work/modele"

if os.path.exists(model_save_dir):
    shutil.rmtree(model_save_dir)

os.makedirs(model_save_dir, exist_ok=True)

model_name = model_save_dir + "/model.pth"
torch.save(model.state_dict(), model_name)
print("Modèle sauvegardé au format .pth dans ",model_name)


######## Sauvegarde des paramètres ########


import torch
import numpy as np
from ML_Perceptron import ML_Perceptron
import json

model.load_state_dict(torch.load(model_name, weights_only=True))
model.eval()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())

# Création du dictionnaire JSON
model_data = {"architecture": [], "parameters": {}}

# Récupérer la structure du réseau
for name, module in model.named_children():
    if isinstance(module, torch.nn.Linear):
        layer_info = {
            "type": "Linear",
            "in_features": module.in_features,
            "out_features": module.out_features,
        }
        model_data["architecture"].append(layer_info)

for name, param in model.named_parameters():
    model_data["parameters"][name] = param.detach().cpu().numpy().tolist()


with open("/home/docker/Work/modele/model_data.json", "w") as f:
    json.dump(model_data, f, indent=4)

print("Modèle sauvegardé sous forme de JSON :/home/docker/Work/modele/model_data.json")
print("\nPour lancer l'inférence sur la Raspberry PI :")
print("-------------------------------\n")
print("1. Copier les dossiers 'modele', 'images_inference' et C'")
print("2. Aller dans le dossier 'C'. Faire les commandes :")
print("make veryclean")
print("make")
print("3. Choisir une image dans images_inference puis lancer l'inférence avec la commande :")
print("./all ../images_inference/image.bmp")
print("\n")