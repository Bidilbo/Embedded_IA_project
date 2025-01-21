import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# PRE TRAITEMENT PAS OBLIGE
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertir en niveaux de gris si nécessaire
    transforms.ToTensor(),  # Convertir en tenseur
])


###### LOADING CUSTOM DATASET ######

dataset_path = "data/dataset_preprocessed"  # Chemin vers le dossier contenant les sous-dossiers
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
print(dataset)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size 
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 5
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Afficher les classes
#print("Classes :", dataset.classes)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    """
    img = X[0].squeeze().numpy()  # Supprimer la dimension des canaux (C) et convertir en numpy
    label = y[0].item()           # Convertir l'étiquette en entier
    plt.imshow(img, cmap="gray") 
    plt.title(f"Label: {label}")
    plt.savefig("./sample_image_custom_dataset.png")"""
    break


######## TRAINING ########

from ML_Perceptron import ML_Perceptron

input_size = 28*28
hidden_size = 128
output_size = 10
model = ML_Perceptron(input_size, hidden_size, output_size).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
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

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "MLP_custom_dataset_pp.pth")
print("Saved PyTorch Model State to MLP_custom_dataset_pp.pth")

# Loading model
model = ML_Perceptron(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load("MLP_custom_dataset_pp.pth", weights_only=True))