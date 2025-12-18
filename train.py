import src.classification_tensorFlow_cut as ctc
import src.classification_tensorFlow_type as ctt
import src.data_prep as dp
import src.metrics as m
#import src.regression_models as rm



# Classification cut
df = dp.preparation_dataset_50k()
df = dp.preparation_dataset_50k()
(X_train, X_test, y_train, y_test, 
 X_train_scaled, X_test_scaled, 
 y_train_cat, y_test_cat, scaler) = ctc.preparation_entrainement(df)
model = ctc.model_tensorFlow(y_train, y_test, X_train_scaled, X_test_scaled, y_train_cat, y_test_cat)
ctc.sauvegarder_modele_tf(model, "tfCut", scaler)