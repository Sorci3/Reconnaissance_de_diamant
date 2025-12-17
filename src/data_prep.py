import pandas as pd


def preparation_dataset_6k_Classification() :
    """
    Fonction qui prépare les données pour le dataset de 6k lignes pour la classification.
    
    Variable d'entré : Aucune

    Fonctionnement : 
    - Load du csv
    - Nettoyage du nom des colonnes pour une bonne comphréhension
    - Mapping
    - Drop des colonnes inutiles
    - Remplacement des vides pour les colonnes restantes
    - Encodage

    Variable de sortie : Dataframe
    """

    df = pd.read_csv("../data/diamonds_dataset.csv") 

    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.replace(' ', '_').str.replace('%', 'pct').str.replace('/', '_per_')

    # Mapping 
    grade_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}
    fluo_map = {'None': 0, 'Faint': 1, 'Medium': 2, 'Strong': 3, 'Very Strong': 4}
    color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
    type_map = {'GIA Lab-Grown' : 0, 'GIA' : 1, 'IGI Lab-Grown' : 2}

    df['Polish'] = df['Polish'].map(grade_map)
    df['Symmetry'] = df['Symmetry'].map(grade_map)
    df['Color'] = df['Color'].map(color_map)
    df['Clarity'] = df['Clarity'].map(clarity_map)
    df['Type'] = df['Type'].map(type_map)

    # Suppression de la colonne Cut et Culet du au trop grand nombre de Nan
    df = df.drop(columns=['Cut']) 
    df = df.drop(columns=['Culet']) 

    # Pour Fluorescence on remplace les vides par 0 
    df['Fluorescence'] = df['Fluorescence'].map(fluo_map).fillna(0)

    # Encodage des variables catégorielles restantes 
    cols_to_dummy = ['Shape', 'Girdle'] 
    df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=True)

    df.dropna(inplace=True)

    return df



def preparation_dataset_50k_Classification() : 
    """
    Fonction qui prépare les données pour le dataset de 50k lignes pour la classification.
    
    Variable d'entré : Aucune

    Fonctionnement : 
    - Load du csv
    - Nettoyage du nom des colonnes pour une bonne comphréhension
    - Mapping
    - Feature Engineering

    Variable de sortie : Dataframe
    """

    df = pd.read_csv("../data/diamonds.csv")
    df = df.drop(df.columns[0], axis=1)

    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.replace(' ', '_').str.replace('%', 'pct').str.replace('/', '_per_')

    # Mapping
    cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

    df['cut'] = df['cut'].map(cut_map)
    df['color'] = df['color'].map(color_map)
    df['clarity'] = df['clarity'].map(clarity_map)

    # Feature Engineering 
    df['volume'] = df['x'] * df['y'] * df['z']
    df['volume'] = df['volume'].replace(0, 0.01) 
    df['table_depth_ratio'] = df['table'] / df['depth']
    df['lw_ratio'] = df['x'] / (df['y'] + 0.001)

    return df



def preparation_dataset_50k_Regression() :
    """
    Fonction qui prépare les données pour le dataset de 50k lignes pour la regression.
    
    Variable d'entré : Aucune

    Fonctionnement : 
    - Load du csv
    - Nettoyage du nom des colonnes pour une bonne comphréhension
    - Mapping

    Variable de sortie : Dataframe
    """

    df = pd.read_csv("../data/diamonds.csv")
    df = df.drop(df.columns[0], axis=1)

    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.replace(' ', '_').str.replace('%', 'pct').str.replace('/', '_per_')

    # Mapping
    cut_map = {'Fair': 0,'Good': 1,'Very Good': 2,'Premium': 3,'Ideal': 4}
    color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

    df['cut'] = df['cut'].map(cut_map)
    df['color'] = df['color'].map(color_map)
    df['clarity'] = df['clarity'].map(clarity_map)


    return df




