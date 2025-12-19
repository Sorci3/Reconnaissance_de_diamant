import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


#####################################################
#              Préparation des données
#####################################################

def preparation_entrainement(df):
    """
    Fonction qui prépare les données pour l'entrainement par un train, test et
    split puis par une standardisation et un to_numerical.

    Paramètre d'entré : df --> dataframe
    Paramètre de sortie : X_train, X_test, y_train, y_test, X_train_scaled,
                          X_test_scaled, y_train_cat, y_test_cat
    """
    target = 'cut'
    X = df.drop(columns=[target])
    y = df[target]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) 
    X_test_scaled = scaler.transform(X_test)

    y_train_cat = to_categorical(y_train, num_classes=5)
    y_val_cat = to_categorical(y_val, num_classes=5)
    y_test_cat = to_categorical(y_test, num_classes=5)

    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train_cat, y_val_cat, y_test_cat, scaler)


#####################################################
#              Modèle Decision Tree
#####################################################

def model_decision_tree(X_train, y_train, X_test, y_test):
    """
    Modèle du decision tree

    Paramètre d'entré : X_train, y_train, X_test, y_test
    Paramètre de sortie : Aucun
    Métriques : Accuracy

    IMPORTANT : Ce modèle étant un decision tree il n'a pas besoin des
    données standardisé
    """
    tree_classifier = DecisionTreeClassifier(random_state=42)
    tree_classifier.fit(X_train, y_train)

    y_pred_train = tree_classifier.predict(X_train)
    y_pred_test = tree_classifier.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print('===========================================================')
    print('Modèle Decision Tree')
    print('===========================================================')
    print(f'Accuracy train : {accuracy_train:.2f}')
    print(f'Accuracy test: {accuracy_test:.2f}')

    return 0


#####################################################
#           Modèle Decision Tree avec Grid Search
#####################################################

def model_decision_tree_grid_search(X_train, y_train, X_test, y_test):
    """
    Modèle du decision tree optimisé avec Grid Search

    Paramètre d'entré : X_train, y_train, X_test, y_test
    Paramètre de sortie : Le meilleur modèle trouvé
    Métriques : Accuracy

    IMPORTANT : Ce modèle étant un decision tree il n'a pas besoin des
    données standardisé
    """
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 15, 20, 30],
        'min_samples_split': [2, 10, 50],
        'min_samples_leaf': [1, 5, 10]
    }

    dt_base = DecisionTreeClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=dt_base,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)
    best_tree = grid_search.best_estimator_

    # y_pred_best = best_tree.predict(X_test)
    # accuracy_best = accuracy_score(y_test, y_pred_best)

    # print('=================================================')
    # print('Modèle Decision Tree avec Grid Search')
    # print('=================================================')
    # print(f'Meilleurs paramètres : {grid_search.best_params_}')
    # print(f'Meilleur score de validation (cv) : {grid_search.best_score_:.4f}')
    # print(f'Accuracy Test (Optimisé) : {accuracy_best:.4f}')

    return best_tree


#####################################################
#           Modèle TensorFlow
#####################################################

def model_tensorFlow(y_train, X_train_scaled, X_val_scaled, y_train_cat, y_val_cat):
    """
    Modèle Tensorflow
    Paramètre d'entré' : y_train, y_test, X_train_scaled, X_test_scaled,
                         y_train_cat, y_test_cat
    Paramètre de sortie : Le modèle

    Fonctionnement
    - Définition du modèle
    - Entrainement
    - Affichage des metrics
    """

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    model.summary()

    # Entrainement
    classes_uniques = np.unique(y_train)
    poids = class_weight.compute_class_weight('balanced', classes=classes_uniques, y=y_train)
    poids_dict = dict(zip(classes_uniques, poids))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    history = model.fit(
        X_train_scaled,
        y_train_cat,
        epochs=50,
        batch_size=64,
        validation_data=(X_val_scaled, y_val_cat), 
        callbacks=[early_stopping, lr_scheduler],
        class_weight=poids_dict,
        verbose=2
    )

    return model


#####################################################
#           Sauvegarde et Chargement
#####################################################



#Tensorflow

def sauvegarder_modele_tf(model, nom_model, scaler):
    """
    Sauvegarde le modèle TensorFlow au format .keras
    """
    dossier = 'model'
    if not os.path.exists(dossier):
        os.makedirs(dossier)
        
    chemin_complet = os.path.join(dossier, f'{nom_model}.keras')
    model.save(chemin_complet)
    joblib.dump(scaler, os.path.join(dossier, f'{nom_model}_scaler.pkl'))
    print(f' Modèle et Scaler sauvegardés dans {dossier}')

def charger_et_tester_modele_tf(nom_model, X_test_scaled, y_test_cat):
    """
    Charge un modèle sauvegardé et effectue une évaluation sur les données de test
    """
    chemin_complet = f'model/{nom_model}.keras'
    
    if not os.path.exists(chemin_complet):
        print(f" Erreur : Le fichier {chemin_complet} n'existe pas.")
        return None

    # Chargement du modèle
    model = tf.keras.models.load_model(chemin_complet)
    print(f" Modèle {nom_model} chargé avec succès.")

    # Évaluation
    loss, acc, prec, rec = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    
    print('=================================================')
    print(f'Résultats du modèle chargé : {nom_model}')
    print('=================================================')
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall   : {rec:.4f}")
    
    return 0


# Decision Tree
def sauvegarder_modele_dt(model, nom_model):
    """ Sauvegarde un modèle Scikit-Learn au format .pkl """
    dossier = 'model'
    if not os.path.exists(dossier):
        os.makedirs(dossier)
    
    chemin = os.path.join(dossier, f'{nom_model}.pkl')
    joblib.dump(model, chemin)
    print(f" Modèle Decision Tree sauvegardé : {chemin}")



def charger_et_tester_modele_dt(nom_model, X_test, y_test):
    """ Charge le modèle .pkl et le teste """
    chemin = f'model/{nom_model}.pkl'
    
    if not os.path.exists(chemin):
        print(f" Erreur : {chemin} introuvable.")
        return
    
    model = joblib.load(chemin)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    
    print('=================================================')
    print(f'Résultats Decision Tree chargé : {nom_model}')
    print('=================================================')
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    return 0