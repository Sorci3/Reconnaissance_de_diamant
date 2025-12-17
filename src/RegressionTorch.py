import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def Regression_Torch_train(X,y,DropOutPercentage = 0.5,layer1 = 256,layer2=128,layer3=64,activationP):
    activation = 0.1

    #Separation des valeurs dans train et test
    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    #Scaling des données
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)
    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test)

    #Recuperation des dimension
    input_dimension = X_train.shape[1]
    output_dimension = y_train.shape[1]

    #Creation des Tensor pour le modele
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    model = nn.Sequential(
        nn.Linear(input_dimension, layer1),
        nn.BatchNorm1d(layer1),
        nn.LeakyReLU(activation),
        nn.Dropout(p),

        nn.Linear(layer1, layer2),
        nn.BatchNorm1d(layer2),
        nn.LeakyReLU(activation),
        nn.Dropout(p),

        nn.Linear(layer2, layer3),
        nn.BatchNorm1d(layer3),
        nn.LeakyReLU(activation),

        nn.Linear(layer3, output_dimension)
    )


