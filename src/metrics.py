import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def torch_regression_metrics(
        model,
        X_tensor,
        y_true_scaled,
        scaler_y,
        dataset_name
):
    """
        Calcule les métriques de performance (RMSE, MAE, R2) pour le modèle.

        Args:
            model (nn.Module): Modèle PyTorch entraîné.
            X_tensor (torch.Tensor): Données d'entrée (scaled).
            y_true_scaled (np.ndarray): Valeurs cibles réelles (scaled).
            scaler_y (StandardScaler): Scaler utilisé pour la cible (pour inverse transform).
            dataset_name (str): Nom du dataset (ex: 'Train' ou 'Test').

        Returns:
            pd.DataFrame: DataFrame contenant les métriques calculées.
        """

    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_tensor).numpy()

    # Retour aux valeurs réelles
    preds_real = scaler_y.inverse_transform(preds_scaled)
    y_true_real = scaler_y.inverse_transform(y_true_scaled)

    results = []
    targets = ['Price ($)', 'Carat (ct)']

    for i, target_name in enumerate(targets):
        y_true = y_true_real[:, i]
        y_pred = preds_real[:, i]

        results.append({
            'Dataset': dataset_name,
            'Cible': target_name,
            'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
            'MAE': round(mean_absolute_error(y_true, y_pred), 2),
            'R2 Score': round(r2_score(y_true, y_pred), 4)
        })
    return pd.DataFrame(results)


def tensorflow_regression_metrics(Y_true, Y_pred, dataset_name):
    """Calcule et formate les métriques pour le rapport."""
    
    # Séparation des cibles pour les calculs
    y_true_price = Y_true[:, 0]
    y_pred_price = Y_pred[:, 0]
    
    y_true_carat = Y_true[:, 1]
    y_pred_carat = Y_pred[:, 1]
    
    results = []
    
    # Métriques pour le prix
    results.append({
        'Dataset': dataset_name,
        'Cible': 'Price ($)',
        'RMSE': np.sqrt(mean_squared_error(y_true_price, y_pred_price)),
        'MAE': mean_absolute_error(y_true_price, y_pred_price),
        'R2 Score': r2_score(y_true_price, y_pred_price)
    })
    
    # Métriques pour le carat
    results.append({
        'Dataset': dataset_name,
        'Cible': 'Carat (ct)',
        'RMSE': np.sqrt(mean_squared_error(y_true_carat, y_pred_carat)),
        'MAE': mean_absolute_error(y_true_carat, y_pred_carat),
        'R2 Score': r2_score(y_true_carat, y_pred_carat)
    })
    
    return pd.DataFrame(results)