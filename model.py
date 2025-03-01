#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:26:22 2024

@author: mridul
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import EsmTokenizer, EsmModel
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class DataProcess:
    def __init__(self, esm2="facebook/esm2_t36_3B_UR50D", device="cuda"): #change depending on the esm2 model
        self.tokenizer = EsmTokenizer.from_pretrained(esm2)
        self.esm_model = EsmModel.from_pretrained(esm2).to(device).eval()
        self.device = device

    # Process sequences in batches -> tokenize them ->  get their embeddings
    def process_in_batches(self, sequences, batch_size=15):
        batched_outputs = []
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True, max_length=1500) # uses 1500 length from the sequence
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.esm_model(**inputs)
            batched_outputs.append(outputs.last_hidden_state.mean(dim=1)) # embeddings are the mean of the output layer columns
        return torch.cat(batched_outputs, dim=0)

    # Load data from a CSV file -> preprocess and tokenize sequences -> extract targets if available
    def load_and_preprocess_data(self, file_path):
        df = pd.read_csv(file_path)
        sequences = df['sequence'].tolist()
        targets = df.get('target', None)
        if targets is not None:
            targets = targets.to_numpy()
        return sequences, targets

    # Split data into training and validation sets
    def train_val_split(self, embeddings, targets, test_size=0.2, random_state=32): #test size here is 20% of the total dataset
        return train_test_split(embeddings, targets, test_size=test_size, random_state=random_state)

    #  DataLoader objects for both training and validation sets
    def create_dataloaders(self, X_train, X_val, y_train, y_val, batch_size=64):
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

'''This is the main model that uses the inputs from the Embeddings created by ESM-2 model.
 I have structures this in such a way that the hyperparameters can be passed as lists, thus allowing to test 
multiple layers, without changing the model code'''

class main_model_class(nn.Module):

    def __init__(self, input_size=2560, hidden_layer_size=[640, 128], activation='tanh', dropout_p=0.0):
        super(main_model_class, self).__init__()
        layers = []
        for i in range(len(hidden_layer_size)):
            layers.append(nn.Linear(input_size if i == 0 else hidden_layer_size[i-1], hidden_layer_size[i])) #linear layer
            if activation == 'tanh': #activation fucntion can be specified by user, default is tanh.
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU()) # activation funcrion
            layers.append(nn.Dropout(dropout_p)) # applying dropout
        layers.append(nn.Linear(hidden_layer_size[-1], 1)) # final layer which outputs only 1 value
        self.model = nn.Sequential(*layers)

    # forward pass
    def forward(self, x):
        return self.model(x)
    
# Train the model 
def train_model(train_loader, val_loader, device, hyperparameters):
    lr = hyperparameters['lr']
    hidden_layer_size = hyperparameters['hidden_layer_size']
    activation = hyperparameters['activation']
    optimizer_name = hyperparameters['optimizer']
    dropout_p = hyperparameters['dropout_p']
    epochs = hyperparameters['epochs']

    model = main_model_class(input_size=2560, hidden_layer_size=hidden_layer_size, activation=activation, dropout_p=dropout_p) # input size is 2560 as the t36 model outputs 2560 features
    model.to(device)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # using mse for loss

    train_losses = [] # lists to keep track of loss values
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        val_losses.append(val_loss)

        # Print training and validation loss for each epoch
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        if val_loss < 11.8: # conditional break, can be changed.
            print("val loss < 11.8 - early stop")
            break

    # plots the train and val loss - this is useful to check if model is underfitting or overfitting
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model # returns model

'''this function predicts and save the results to a CSV file
I have made this function in such a way that if you already have the embeddings calculated, you dont need to calculate them again
just make sure that you have these file names on  your path
this saves a tonne of computational time 
in case you want to run this i have pre-computed embeddings at this link which you can download -
https://drive.google.com/drive/folders/12TDMgEKjjCJd3WSoMhRNWnfcEQBoGao_?usp=sharing 
'''
def predict_and_save(model, device, test_file_path='test.csv', predictions_file_path='predictions.csv', test_embeddings_file='1500_t36_embeddings_test.npy'):
    df = pd.read_csv(test_file_path)
    sequences = df['sequence'].tolist()
    ids = df['id'].tolist()

    # Check if test embeddings already exist, if they do, then load that.
    if os.path.exists(test_embeddings_file):
        print("Loading existing test embeddings.")
        embeddings = np.load(test_embeddings_file)
    else:
        print("Generating new test embeddings.")
        embeddings = main_model2.process_in_batches(sequences).cpu().numpy()
        np.save(test_embeddings_file, embeddings)

    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(embeddings, dtype=torch.float32).to(device)).squeeze().cpu().numpy()

    pd.DataFrame({'id': ids, 'target': predictions}).to_csv(predictions_file_path, index=False)
    print(f"Predictions saved to {predictions_file_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # checks for gpu
    main_model2 = DataProcess(device=device)

    # Load and preprocess training data
    sequences, targets = main_model2.load_and_preprocess_data('train.csv')

    if os.path.exists('train_1500_t36_embeddings.npy'):
        embeddings = np.load('train_1500_t36_embeddings.npy')
    else:
        print("Checked for embeddings : they are not available : Generating new training embeddings.")
        embeddings = main_model2.process_in_batches(sequences).cpu().numpy()
        np.save('train_1500_t36_embeddings.npy', embeddings)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)

    X_train, X_val, y_train, y_val = main_model2.train_val_split(embeddings, targets)
    train_loader, val_loader = main_model2.create_dataloaders(X_train, X_val, y_train, y_val)

    #these hyperparameters gave me the best result for this model and data
    hyperparameters = {
        'lr': 0.0001/4,  
        'hidden_layer_size': [1280, 100], 
        'activation': 'tanh',  
        'optimizer': 'Adam',  
        'dropout_p': 0.2418326688229436/3,  
        'epochs': 340 
    }

    model = train_model(train_loader, val_loader, device, hyperparameters)
    predict_and_save(model, device)
 
