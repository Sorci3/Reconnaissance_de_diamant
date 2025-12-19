import src.classification_tensorFlow_cut as ctc
import src.classification_tensorFlow_type as ctt
import src.data_prep as dp
import src.metrics as m
import pickle
import torch
import src.metrics as m
import src.regression_models as rm

def run_regression_training():
    df = dp.preparation_dataset_50k()
    
    print("|| Entraînement PyTorch ||")
    model_torch, artifacts_torch, _ = rm.train_torch_regression(
        df, num_epochs=50
    )
    #Sauvegarde du modèle pour les tests
    torch.save(model_torch, "model/torch_regression_full.pth")
    with open("model/torch_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts_torch, f)
    print("Modèle PyTorch sauvegardé.")

    print("|| Entraînement TensorFlow ||")
    model_tf, history, artifacts_tf = rm.train_tensorflow_regression(
        df, epochs=20
    )

    #Sauvegarde du modèle pour les tests
    model_tf.save("model/tf_regression_model.keras")
    with open("model/tf_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts_tf, f)
    print("Modèle TensorFlow sauvegardé.")


def run_sklearn_baseline():
    print("||  Entraînement Baseline (Linear Regression) || ")
    df = dp.preparation_dataset_50k()
    
    # Entraînement
    pipeline, X_test, y_test = rm.train_sklearn_pipeline(df)
    
    # Sauvegarde du modèle
    path_model = "model/sklearn_pipeline.pkl"
    with open(path_model, "wb") as f:
        pickle.dump(pipeline, f)
        
    # Sauvegarde des données de test pour l'évaluation
    path_data = "model/sklearn_test_data.pkl"
    with open(path_data, "wb") as f:
        pickle.dump((X_test, y_test), f)
        
    print(f"Baseline sauvegardée.")

if __name__ == "__main__":

    run_regression_training()
    run_sklearn_baseline()
    
    
    
    # Classification cut

    ##Decision Tree
    df = dp.preparation_dataset_50k()
    X_train, _, X_test, y_train, _, y_test, _, _, _, _, _, _, _ = ctc.preparation_entrainement(df)
    best_model = ctc.model_decision_tree_grid_search(X_train, y_train, X_test, y_test)
    ctc.sauvegarder_modele_dt(best_model, "dtCutOptimized")

    ##Tensorflow
    df = dp.preparation_dataset_50k()
    (X_train, X_val, X_test, y_train, y_val, y_test, 
    X_train_scaled, X_val_scaled, X_test_scaled, 
    y_train_cat, y_val_cat, y_test_cat, scaler) = ctc.preparation_entrainement(df)
    model = ctc.model_tensorFlow(y_train, X_train_scaled, X_val_scaled, y_train_cat, y_val_cat)
    ctc.sauvegarder_modele_tf(model, "tfCut", scaler)


    # Classification Type

    ##Decision Tree
    df = dp.preparation_dataset_6k_Classification()
    X_train, _, X_test, y_train, _, y_test, _, _, _, _, _, _, _ = ctt.preparation_entrainement(df)
    best_model = ctt.model_decision_tree_grid_search(X_train, y_train, X_test, y_test)
    ctt.sauvegarder_modele_dt(best_model, "dtTypeOptimized")

    ##TensorFlow
    df = dp.preparation_dataset_6k_Classification()
    (X_train, X_val, X_test, y_train, y_val, y_test, 
    X_train_scaled, X_val_scaled, X_test_scaled, 
    y_train_cat, y_val_cat, y_test_cat, scaler) = ctt.preparation_entrainement(df)
    model = ctt.model_tensorFlow(y_train, X_train_scaled, X_val_scaled, y_train_cat, y_val_cat)
    ctt.sauvegarder_modele_tf(model, "tfType", scaler)

    