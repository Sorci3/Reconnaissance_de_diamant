import src.classification_tensorFlow_cut as ctc
import src.data_prep as dp
import joblib
from tensorflow.keras.utils import to_categorical

# Classification Cut
nom_du_modele = "tfCut"
    
df = dp.preparation_dataset_50k()
X = df.drop(columns=['cut'])
y = df['cut']

scaler = joblib.load(f"model/{nom_du_modele}_scaler.pkl")

X_test_scaled = scaler.transform(X) 
y_test_cat = to_categorical(y, num_classes=5)

ctc.charger_et_tester_modele_tf(nom_du_modele, X_test_scaled, y_test_cat)