# Deep Learning

### Contributeurs
Louis Maillet
Herbreteau Mathis
Le Pottier Mathias
Pasquier Samuel

### Contexte
Nous sommes des bijoutiers et nous souhaitons pouvoir garantir l'authenticité des diamants apporté par nos clients. Nous souhaitons également les racheters au prix juste. Pour ce faire nous avons développé un système permettant de prédire la coupe, le type, le prix et le carat pour les diamants n'ayant jamais été authentifiés.

### Run le projet
**Soyez sur d'être à la racine**

Implémenter les requirements
```Bash
pip install requirements.txt
```

Lancer le fichier qui train nos modèles
```Bash
python train.py
```

Lancer le fichier qui test nos modèles
```Bash
python app.py
```

### Descriptions des fichiers et dossiers présents dans src
classification_tensorFlow_cut.py : contient 3 modèles pour la classification sur la feature 'cut'.

classification_tensorFlow_cut.py : contient 3 modèles pour la classification sur la feature 'Type'.

data_prep.py : Le fichier contient les fonctions permettant de préparer les données pour les différents modèles.

regression_models : Le fichier contient 3 modèles pour la régréssion sur le prix le carat.

Le dossier model contient les modèles que nous avons entrainés.

