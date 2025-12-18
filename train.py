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
    
    print(">> Entraînement PyTorch...")
    model_torch, artifacts_torch, _ = rm.train_torch_regression(
        df, num_epochs=50
    )

    torch.save(model_torch, "model/torch_regression_full.pth")
    with open("model/torch_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts_torch, f)
    print("Modèle PyTorch sauvegardé dans 'saved_models/'.")

    print(">> Entraînement TensorFlow...")
    model_tf, history, artifacts_tf = rm.train_tensorflow_regression(
        df, epochs=20
    )

    model_tf.save("model/tf_regression_model.keras")
    with open("model/tf_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts_tf, f)
    print("Modèle TensorFlow sauvegardé dans 'saved_models/'.")


if __name__ == "__main__":

    run_regression_training()

    # Classification cut
    df = dp.preparation_dataset_50k()
    df = dp.preparation_dataset_50k()
    (X_train, X_test, y_train, y_test, 
     X_train_scaled, X_test_scaled, 
     y_train_cat, y_test_cat, scaler) = ctc.preparation_entrainement(df)
    model = ctc.model_tensorFlow(y_train, y_test, X_train_scaled, X_test_scaled, y_train_cat, y_test_cat)
    ctc.sauvegarder_modele_tf(model, "tfCut", scaler)