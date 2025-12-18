import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import src.classification_tensorFlow_cut as ctc
import src.data_prep as dp
import joblib
from tensorflow.keras.utils import to_categorical


import torch
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import src.metrics
import src.regression_models

def evaluate_tensorflow_model():
    print("\n--- Évaluation Modèle TensorFlow (Régression) ---")
    
    path_model = "model/tf_regression_model.keras"
    path_artifacts = "model/tf_artifacts.pkl"
    
    if not os.path.exists(path_model):
        print("Erreur : Modèle non trouvé. Lancez train.py d'abord.")
        return

    model = tf.keras.models.load_model(path_model)
    
    with open(path_artifacts, "rb") as f:
        artifacts = pickle.load(f)
        
    X_test = artifacts['X_test']
    y_test_scaled = artifacts['y_test']
    scaler_y = artifacts['scaler_y']
    
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform pour avoir les vrais Prix et Carats
    y_true_real = scaler_y.inverse_transform(y_test_scaled)
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
    
    #Calcul des métriques via metrics.py
    df_metrics = src.metrics.tensorflow_regression_metrics(
        Y_true=y_true_real,
        Y_pred=y_pred_real,
        dataset_name='Test Set'
    )
    print("Résultats Régression TensorFlow :")
    print(df_metrics.to_string(index=False))


def evaluate_torch_model():
    print("\n--- Évaluation Modèle PyTorch (Régression) ---")
    
    path_model = "model/torch_regression_full.pth"
    path_artifacts = "model/torch_artifacts.pkl"


    # 1. Chargement du modèle complet
    model = torch.load(path_model, weights_only=False)
    model.eval()
    
    # 2. Chargement des artifacts (X_test, scalers...)
    with open(path_artifacts, "rb") as f:
        artifacts = pickle.load(f)
    
    # 3. Calcul des métriques via metrics.py
    # Note: artifacts['X_test_tensor'] contient déjà les données scalées de test
    df_metrics = src.metrics.torch_regression_metrics(
        model=model,
        X_tensor=artifacts['X_test_tensor'],
        y_true_scaled=artifacts['y_test_scaled'],
        scaler_y=artifacts['scaler_y'],
        dataset_name='Test Set'
    )
    
    print(df_metrics.to_string(index=False))
    
    # Exemple de prédiction unitaire (Optionnel)
    print("\nTest de prédiction unitaire :")
    prix, carat = src.regression_models.torch_regression_prediction(
        model, artifacts['scaler_x'], artifacts['scaler_y'],
        color_str='E', clarity_str='VS1', depth=61.0, table=57.0, x=5.0, y=5.0, z=3.0
    )
    print(f"Prediction -> Prix: ${prix:.2f}, Carat: {carat:.4f}")


def evaluate_sklearn_model():
    print("\n--- Évaluation Baseline Scikit-Learn ---")
    
    path_model = "model/sklearn_pipeline.pkl"
    path_data = "model/sklearn_test_data.pkl"

    #Chargement du modèle
    with open(path_model, "rb") as f:
        pipeline = pickle.load(f)
        
    with open(path_data, "rb") as f:
        X_test, y_test = pickle.load(f)
        
    #Prédiction
    y_pred = pipeline.predict(X_test)
    
    #Evaluation des résultats 
    df_metrics = src.metrics.tensorflow_regression_metrics(
        Y_true=y_test.values, 
        Y_pred=y_pred,
        dataset_name='Test Set (Sklearn)'
    )
    
    print(df_metrics.to_string(index=False))

if __name__ == "__main__":
    
    
    evaluate_tensorflow_model()
    evaluate_torch_model()
    evaluate_sklearn_model()
    
    # Classification Cut
    nom_du_modele = "tfCut"
    
    df = dp.preparation_dataset_50k()
    X = df.drop(columns=['cut'])
    y = df['cut']

    scaler = joblib.load(f"model/{nom_du_modele}_scaler.pkl")

    X_test_scaled = scaler.transform(X) 
    y_test_cat = to_categorical(y, num_classes=5)

    ctc.charger_et_tester_modele_tf(nom_du_modele, X_test_scaled, y_test_cat)