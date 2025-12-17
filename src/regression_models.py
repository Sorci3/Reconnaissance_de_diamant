import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}


def train_torch_regression(
        dataframe,
        feature_cols = None,
        target_cols = None,
        dropout_percentage = 0.5,
        layer1 = 256,
        layer2 = 128,
        layer3 = 64,
        num_epochs = 300
):
    """
        Entraîne un réseau de neurones PyTorch pour une tâche de régression.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données.
            feature_cols (List[str], optional): Liste des colonnes de caractéristiques.
                Défaut à None (utilise une liste prédéfinie).
            target_cols (List[str], optional): Liste des colonnes cibles.
                Défaut à None (['price', 'carat']).
            dropout_rate (float): Pourcentage de Dropout (entre 0 et 1).
            layer1_size (int): Taille de la première couche cachée.
            layer2_size (int): Taille de la deuxième couche cachée.
            layer3_size (int): Taille de la troisième couche cachée.
            num_epochs (int): Nombre d'époques d'entraînement.
            learning_rate (float): Taux d'apprentissage pour l'optimiseur.

        Returns:
            Tuple containing:
                - model (nn.Module): Le modèle entraîné.
                - data_artifacts (dict): Dictionnaire contenant les tenseurs et scalers
                  (X_train, X_test, y_train, y_test, scaler_X, scaler_y).
                - model_summary (str): Résumé du modèle via torchsummary.
        """

    if feature_cols is None:
        feature_cols = ['color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
    if target_cols is None:
        target_cols = ['price', 'carat']

    activation_slope = 0.1

    X = dataframe[feature_cols]
    y = dataframe[target_cols]

    #Separation des valeurs dans train et test
    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    #Scaling des données
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)
    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test)

    #Récupération des dimensions
    input_dimension = X_train.shape[1]
    output_dimension = y_train.shape[1]

    #Création des Tensors pour le modèle
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)


    model = nn.Sequential(
        nn.Linear(input_dimension, layer1),
        nn.BatchNorm1d(layer1),
        nn.LeakyReLU(activation_slope),
        nn.Dropout(dropout_percentage),

        nn.Linear(layer1, layer2),
        nn.BatchNorm1d(layer2),
        nn.LeakyReLU(activation_slope),
        nn.Dropout(dropout_percentage),

        nn.Linear(layer2, layer3),
        nn.BatchNorm1d(layer3),
        nn.LeakyReLU(activation_slope),

        nn.Linear(layer3, output_dimension)
    )

    criterion = (nn.MSELoss())
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=20, factor=0.5
    )


    for epoch in range(num_epochs):
        model.train()

        # Forward Pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Évaluation du modèle avec le test
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)

        # Update du scheduler
        scheduler.step(test_loss)

        #Affichage
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}')

    data_artifacts = {
        'X_train_tensor': X_train_tensor,
        'X_test_tensor': X_test_tensor,
        'y_train_scaled': y_train,
        'y_test_scaled': y_test,
        'scaler_x': scaler_X,
        'scaler_y': scaler_y
    }

    model_summary = summary(model, input_size=input_dimension, output_size=output_dimension)

    return model, data_artifacts, model_summary


def torch_regression_prediction(
        model,
        scaler_X,scaler_y,
        color_str,
        clarity_str,
        depth,
        table,
        x, y, z
):
    """
        Predit le Prix et le Carat d'un diamant à partir de ses caractéristiques.

        Args:
            model (nn.Module): Le modèle entraîné.
            scaler_x (StandardScaler): Scaler ajusté sur les features d'entraînement.
            scaler_y (StandardScaler): Scaler ajusté sur les cibles d'entraînement.
            color_str (str): Couleur (ex: 'E', 'J', 'D').
            clarity_str (str): Clarté (ex: 'VS1', 'IF').
            depth, table, x, y, z (float): Dimensions et propriétés géométriques.

        Returns:
            Union[Tuple[float, float], str]: (Prix estimé, Carat estimé) ou message d'erreur.
        """

    if color_str not in color_map or clarity_str not in clarity_map:
        return "Erreur : valeur non valide"

    color_val = color_map[color_str]
    clarity_val = clarity_map[clarity_str]

    #Création d'un vecteur d'entrée
    raw_input = np.array(
        [[color_val, clarity_val, depth, table, x, y, z]],
        dtype=np.float32
    )

    #Normalisation (Scaling)
    scaled_input = scaler_X.transform(raw_input)

    #Conversion en Tensor PyTorch
    tensor_input = torch.tensor(scaled_input)

    #Prédiction
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(tensor_input)

    #Dénormalisation
    prediction_real = scaler_y.inverse_transform(prediction_scaled.numpy())

    #Extraction des résultats
    prix_estime = prediction_real[0][0]
    carat_estime = prediction_real[0][1]

    return prix_estime, carat_estime




