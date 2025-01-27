import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tqdm import tqdm 
import pandas as pd

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using {device} device.")


train_dir = os.path.join("..","data", "Training")
test_dir = os.path.join("..","data", "Testing")



class ConvertImage:
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert("RGB")
        return img    


transform = transforms.Compose(
    [
        ConvertImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1855, 0.1856, 0.1856], std=[0.2003, 0.2003, 0.2003]),
    ]
)

batch_size = 32
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)


# create dataloader
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



def train_model(dataloader, device, optimizer, loss_fn, model):
    
    model.train()
    training_loss = 0.0
    for data, label in tqdm(dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)
        
        output = model(data)
        loss = loss_fn(output, label)
        
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item() * data.size(0)
        
    return training_loss / len(dataloader.dataset)


def predict(model, dataloader, device):
    model.eval()
    prob = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            data = data.to(device)
            output = model(data)
            out_prob = nn.functional.softmax(output, dim=1)
            
            prob = torch.cat((prob, out_prob), dim=0)
    return prob

def loss_accuracy(model, dataloader, loss_fn, device):
    total_loss = 0
    total_correct = 0

    model.to(device)
    model.eval()
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            data = data.to(device)
            output = model(data)
            
            label = label.to(device)
            loss = loss_fn(output, label)
            total_loss += loss.data.item() * data.size(0)
            
            correct = torch.eq(torch.argmax(output, dim=1), label)
            total_correct += torch.sum(correct).item()
            
    n_observations = dataloader.batch_size * len(dataloader)
    accuracy = total_correct / n_observations
    average_loss = total_loss / n_observations
    
    return average_loss, accuracy


def early_stopping(validation_loss, best_val_loss, counter):
    """Function that implements Early Stopping"""

    stop = False

    if validation_loss < best_val_loss:
        counter = 0
    else:
        counter += 1

    if counter >= 5:
        step = True

    return counter, stop


def checkpointing(validation_loss, best_val_loss, model, optimizer, save_path):
    if validation_loss < best_val_loss:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": validation_loss,
            },
            save_path,
        )

def train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=20,
    device="cpu",
    scheduler=None,
    checkpoint_path=None,
    early_stopping=None,
):
    # Track the model progress over epochs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []

    # Create the trackers if needed for checkpointing and early stopping
    best_val_loss = float("inf")
    early_stopping_counter = 0

    print("Model evaluation before start of training...")
    # Test on training set
    train_loss, train_accuracy = loss_accuracy(model, train_loader, loss_fn, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    # Test on validation set
    validation_loss, validation_accuracy = loss_accuracy(model, val_loader, loss_fn, device)
    val_losses.append(validation_loss)
    val_accuracies.append(validation_accuracy)

    for epoch in range(1, epochs + 1):
        print("\n")
        print(f"Starting epoch {epoch}/{epochs}")

        # Train one epoch
        train_model(train_loader, device, optimizer, loss_fn, model)

        # Evaluate training results
        train_loss, train_accuracy = loss_accuracy(model, train_loader, loss_fn, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Test on validation set
        validation_loss, validation_accuracy = loss_accuracy(model, val_loader, loss_fn, device)
        val_losses.append(validation_loss)
        val_accuracies.append(validation_accuracy)

        print(f"Epoch: {epoch}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Training accuracy: {train_accuracy*100:.4f}%")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation accuracy: {validation_accuracy*100:.4f}%")

        # # Log the learning rate and have the scheduler adjust it
        lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(lr)
        if scheduler:
            scheduler.step()

        # Checkpointing saves the model if current model is better than best so far
        if checkpoint_path:
            checkpointing(
                validation_loss, best_val_loss, model, optimizer, checkpoint_path
            )

        # Early Stopping
        if early_stopping:
            early_stopping_counter, stop = early_stopping(
                validation_loss, best_val_loss, early_stopping_counter
            )
            if stop:
                print(f"Early stopping triggered after {epoch} epochs")
                break

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss

    return (
        learning_rates,
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        epoch,
    )



model = nn.Sequential(
    # Convolutional Layer 1
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
    # Convolutional Layer 2
    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
    # Convolutional Layer 3
    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
    # Flatten Layer
    nn.Flatten(),
    # Fully Connected Layers
    nn.Linear(in_features=16 * 28 * 28, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=4)
)



optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the learning rate scheduler (StepLR, decreases LR every 10 epochs)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = nn.CrossEntropyLoss()

epochs_to_train = 20

train_results = train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=epochs_to_train,
    device=device,
    scheduler=scheduler,
    checkpoint_path="self_model.pth",
    early_stopping=early_stopping,
)

(
    learning_rates_self,
    train_losses_self,
    valid_losses_self,
    train_accuracies_self,
    valid_accuracies_self,
    epochs,
) = train_results



