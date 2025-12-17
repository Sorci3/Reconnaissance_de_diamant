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