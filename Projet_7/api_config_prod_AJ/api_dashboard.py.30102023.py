import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors

import xgboost as xgb

from joblib import load

import shap

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

@st.cache_data
def load_data():

    # On charge les données
    data_train = pd.read_csv("app_train.csv")
    data_train.drop("Unnamed: 0", axis=1, inplace=True)
    data_test = pd.read_csv("app_test.csv")
    data_test.drop("Unnamed: 0", axis=1, inplace=True)
    data_train_prepared = pd.read_csv("app_train_prepared.csv")
    data_train_prepared.drop("Unnamed: 0", axis=1, inplace=True)
    data_test_prepared = pd.read_csv("app_test_prepared.csv")
    data_test_prepared.drop("Unnamed: 0", axis=1, inplace=True)

    return data_train, data_test, data_train_prepared, data_test_prepared

# @st.cache_data
# def load_xgboost_old():

#     # clf_xgb = load("modele.pickle")
#     import xgboost as xgb

#     # Chargez le modèle XGBoost depuis l'ancienne version
#     old_model = xgb.Booster(model_file='xgboost.pickle')

#     # Sauvegardez le modèle dans la nouvelle version
#     old_model.save_model('xgboostv2.pickle')

#     # # Chargement du modèle avec XGBoost
#     # clf_xgb = xgb.Booster(model_file='modele_xgboost.model')

#     return clf_xgb


# @st.cache_data
def load_knn(df_train):

    knn = entrainement_knn(df_train)
    print("Training knn done")

    return knn

#@st.cache_data()
#def load_logo():
    # Construction de la sidebar
    # Chargement du logo
#    logo = Image.open("logo.png") 
    
#    return logo

@st.cache_data
def load_infos_gen(data_train):

    # Requête permettant de récupérer :
    # Le nombre de lignes de crédits existants dans la base
    # Le revenus moyens des clients
    # Le montant moyen des crédits existants
    lst_infos = [data_train.shape[0],
                 round(data_train["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data_train["AMT_CREDIT"].mean(), 2)]

    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]

    targets = data_train["TARGET"].value_counts()

    return nb_credits, rev_moy, credits_moy, targets


def identite_client(data_test, id):

    data_client = data_test[data_test["SK_ID_CURR"] == int(id)]
    print("shape wd93200wd " , data_client.shape)
    print(data_client)
    print(data_client.columns)

    return data_client

# @st.cache_data
def infos_client_id(data_test, id):
    data_client = data_test[data_test["SK_ID_CURR"] == int(id)]

    dict_infos = {
        "status": [data_client["NAME_FAMILY_STATUS"].item(), "-"],
        "nb_enfant": [data_client["CNT_CHILDREN"].item(), "-"],
        "age": [int(data_client["DAYS_BIRTH"].values / -365), "-"],
        "revenus": [data_client["AMT_INCOME_TOTAL"].item(), "-"],
        "montant_credit": [data_client["AMT_CREDIT"].item(), "-"],
        "annuites": [data_client["AMT_ANNUITY"].item(), "-"],
        "montant_bien": [data_client["AMT_GOODS_PRICE"].item(), "-"]
    }
    # # Ajoutez une ligne vide au dictionnaire avec des espaces ou des tirets
    # dict_infos2 = {
    #     "status": '-',
    #     "nb_enfant": '-',
    #     "age": '-',
    #     "revenus": '-',
    #     "montant_credit": '-',
    #     "annuites": '-',
    #     "montant_bien": '-',
    # }

    # df = pd.DataFrame.from_dict(dict_infos, orient="index")
    # df = pd.DataFrame.from_dict(dict_infos, orient="index").rename(columns={0: ''}.rename(columns={1: ''}
    df = pd.DataFrame.from_dict(dict_infos, orient="index").rename(columns={0: '', 1: ''})

    # Créez un DataFrame en utilisant ces dictionnaires
    # df = pd.DataFrame([dict_infos, dict_infos2])
    # # Créez une ligne vide avec des espaces dans les colonnes
    # empty_row = pd.Series(['_' for _ in range(len(df.columns))], name='')

    # # Ajoutez la première ligne vide à la fin du DataFrame
    # df = df.append(empty_row)

    # # Créez une deuxième ligne vide avec des espaces
    # empty_row2 = pd.Series(['' for _ in range(len(df.columns))], name='')

    # # Ajoutez la deuxième ligne vide à la fin du DataFrame
    # df = df.append(empty_row2)
    df = df.T

    return df
    

@st.cache_data
def load_age_population(data_train):
    
    data_age = round((data_train["DAYS_BIRTH"] / -365), 2)

    return data_age

@st.cache_data
def load_revenus_population(data_train):
    
    # On supprime les outliers qui faussent le graphique de sortie
    # Cette opération supprime un peu moins de 300 lignes sur une
    # population > 300000...
    df_revenus = data_train[data_train["AMT_INCOME_TOTAL"] < 700000]
    
    df_revenus["tranches_revenus"] = pd.cut(df_revenus["AMT_INCOME_TOTAL"], bins=20)
    df_revenus = df_revenus[["AMT_INCOME_TOTAL", "tranches_revenus"]]
    df_revenus.sort_values(by="AMT_INCOME_TOTAL", inplace=True)

    print(df_revenus)
    
    data_revenus = df_revenus["AMT_INCOME_TOTAL"]

    return data_revenus

def load_prediction(data_test, test, id, clf):
    
    print("Analyse data_test :")
    print(data_test.shape)
    print(data_test[data_test["SK_ID_CURR"] == int(id)])
     
    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    print(index[0])
    print(test)
   
    data_client = test.iloc[index[0]]

    print(data_client)

    prediction = clf.predict_proba(data_client)

    prediction = prediction[0].tolist()

    print(prediction)

    return prediction[1]

def load_voisins(data_train, data_test, test, id, mdl):
    
    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    index = index[0]
    print(index)

    data_client = pd.DataFrame(test.iloc[index]).T

    print("Analyse :")
    print("Shape data_client :", data_client.shape)
    print(data_client)

    print("Recherche dossiers en cours...")
    distances, indices = mdl.kneighbors(data_client)

    print("indices")
    print(indices)
    print("distances")
    print(distances)

    df_voisins = data_train.iloc[indices[0], :].copy()
    
    return df_voisins

def entrainement_knn(df):
    print("Entrainement knn en cours...")
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(df)
    return knn  

def train_xgboost_classifier(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.80, test_size=0.20, stratify=Y, random_state=123, shuffle=True)
    
    # Vous pouvez utiliser un autre remplisseur (imputer) si nécessaire, ou ajouter des étapes de prétraitement supplémentaires.
    
    # Convertissez vos données en DMatrix, le format de données spécifique pour XGBoost
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    # Paramètres de l'algorithme XGBoost (vous pouvez les ajuster en fonction de vos besoins)
    params = {
        'objective': 'binary:logistic',  # Classification binaire
        'max_depth': 3,  # Profondeur maximale de l'arbre
        'eta': 0.1,  # Taux d'apprentissage
        'eval_metric': 'logloss'  # Métrique d'évaluation
    }

    # Entraînez le modèle XGBoost
    num_round = 100  # Nombre d'itérations (vous pouvez ajuster cela)
    model = xgb.train(params, dtrain, num_round)
    
    return model, dtest, dtrain, Y_train, Y_test


def entrainement_xgboost(data_train, data_test):
    print("Entrainement xgboost en cours...")
    data_train, data_test = load_and_preprocess_data(data_train, data_test)
    X = data_train
    Y = data_train['TARGET']
    clf_xgb, X_test, X_train, Y_train, Y_test = train_xgboost_classifier(X, Y)
    return clf_xgb  
# @st.cache_data
def load_xgboost(data_train, data_test):

    clf_xgb = entrainement_xgboost(data_train, data_test)
    print("Training xgboost done")

    return clf_xgb
@st.cache_data
def load_prediction_with_shap(data_test, test, id, clf):
    # Initialisation de SHAP
    shap.initjs()

    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values
    index = index[0]
    data_client = test.iloc[index]
    prediction = clf.predict_proba(data_client)
    prediction = prediction[0].tolist()

    # Créez un objet explainer SHAP
    explainer = shap.Explainer(clf)
    shap_values = explainer.shap_values(data_client)

    return prediction[1], shap_values
    #

def features_engineering(data_train, data_test):

    # Cette fonction regroupe toutes les opérations de features engineering
    # mises en place sur les sets train & test

    #############################################
    # LABEL ENCODING
    #############################################
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in data_train:
        if data_train[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(data_train[col].unique())) <= 2:
                # Train on the training data
                le.fit(data_train[col])
                # Transform both training and testing data
                data_train[col] = le.transform(data_train[col])
                data_test[col] = le.transform(data_test[col])
                
                # Keep track of how many columns were label encoded
                le_count += 1

    ############################################
    # ONE HOT ENCODING
    ############################################
    # one-hot encoding of categorical variables
    data_train = pd.get_dummies(data_train)
    data_test = pd.get_dummies(data_test)

    train_labels = data_train['TARGET']
    # Align the training and testing data, keep only columns present in both dataframes
    data_train, data_test = data_train.align(data_test, join = 'inner', axis = 1)
    # Add the target back in
    data_train['TARGET'] = train_labels

    ############################################
    # VALEURS ABERRANTES
    ############################################
    # Create an anomalous flag column
    data_train['DAYS_EMPLOYED_ANOM'] = data_train["DAYS_EMPLOYED"] == 365243
    # Replace the anomalous values with nan
    data_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    data_test['DAYS_EMPLOYED_ANOM'] = data_test["DAYS_EMPLOYED"] == 365243
    # Replace the boolean column by numerics values 
    data_test["DAYS_EMPLOYED_ANOM"] = data_test["DAYS_EMPLOYED_ANOM"].astype("int")


    data_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

    # Traitement des valeurs négatives
    data_train['DAYS_BIRTH'] = abs(data_train['DAYS_BIRTH'])

    ############################################
    # CREATION DE VARIABLES
    ############################################
    # CREDIT_INCOME_PERCENT: the percentage of the credit amount relative to a client's income
    # ANNUITY_INCOME_PERCENT: the percentage of the loan annuity relative to a client's income
    # CREDIT_TERM: the length of the payment in months (since the annuity is the monthly amount due
    # DAYS_EMPLOYED_PERCENT: the percentage of the days employed relative to the client's age

    # Dans cet état d'esprit, nous pouvons créer quelques fonctionnalités qui tentent de capturer ce que nous pensons
    # peut être important pour savoir si un client fera défaut sur un prêt.
    # Ici, je vais utiliser cinq fonctionnalités inspirées de ce script d'Aguiar :

    # CREDIT_INCOME_PERCENT : le pourcentage du montant du crédit par rapport aux revenus d'un client
    # ANNUITY_INCOME_PERCENT : le pourcentage de la rente du prêt par rapport aux revenus d'un client
    # CREDIT_TERM : la durée du versement en mois (puisque la rente est le montant mensuel dû
    # DAYS_EMPLOYED_PERCENT : le pourcentage de jours employés par rapport à l'âge du client

    data_train_domain = data_train.copy()
    data_test_domain = data_test.copy()

    # data_train_domain['CREDIT_INCOME_PERCENT'] = data_train_domain['AMT_CREDIT'] / data_train_domain['AMT_INCOME_TOTAL']
    # data_train_domain['ANNUITY_INCOME_PERCENT'] = data_train_domain['AMT_ANNUITY'] / data_train_domain['AMT_INCOME_TOTAL']
    # data_train_domain['CREDIT_TERM'] = data_train_domain['AMT_ANNUITY'] / data_train_domain['AMT_CREDIT']
    # data_train_domain['DAYS_EMPLOYED_PERCENT'] = data_train_domain['DAYS_EMPLOYED'] / data_train_domain['DAYS_BIRTH']

    # data_test_domain['CREDIT_INCOME_PERCENT'] = data_test_domain['AMT_CREDIT'] / data_test_domain['AMT_INCOME_TOTAL']
    # data_test_domain['ANNUITY_INCOME_PERCENT'] = data_test_domain['AMT_ANNUITY'] / data_test_domain['AMT_INCOME_TOTAL']
    # data_test_domain['CREDIT_TERM'] = data_test_domain['AMT_ANNUITY'] / data_test_domain['AMT_CREDIT']
    # data_test_domain['DAYS_EMPLOYED_PERCENT'] = data_test_domain['DAYS_EMPLOYED'] / data_test_domain['DAYS_BIRTH']
    
    # Calcul de nouvelles caractéristiques basées sur les données du jeu d'entraînement

    # Pourcentage du crédit par rapport au revenu total
    data_train_domain['CREDIT_INCOME_PERCENT'] = data_train_domain['AMT_CREDIT'] / data_train_domain['AMT_INCOME_TOTAL']

    # Pourcentage de l'annuité (paiement mensuel du crédit) par rapport au revenu total
    data_train_domain['ANNUITY_INCOME_PERCENT'] = data_train_domain['AMT_ANNUITY'] / data_train_domain['AMT_INCOME_TOTAL']

    # Terme du crédit : rapport de l'annuité au montant du crédit
    data_train_domain['CREDIT_TERM'] = data_train_domain['AMT_ANNUITY'] / data_train_domain['AMT_CREDIT']

    # Pourcentage des jours d'emploi par rapport à l'âge en jours (une mesure de la stabilité financière)
    data_train_domain['DAYS_EMPLOYED_PERCENT'] = data_train_domain['DAYS_EMPLOYED'] / data_train_domain['DAYS_BIRTH']

    # Calcul de nouvelles caractéristiques basées sur les données du jeu de test

    # Pourcentage du crédit par rapport au revenu total
    data_test_domain['CREDIT_INCOME_PERCENT'] = data_test_domain['AMT_CREDIT'] / data_test_domain['AMT_INCOME_TOTAL']

    # Pourcentage de l'annuité par rapport au revenu total
    data_test_domain['ANNUITY_INCOME_PERCENT'] = data_test_domain['AMT_ANNUITY'] / data_test_domain['AMT_INCOME_TOTAL']

    # Terme du crédit : rapport de l'annuité au montant du crédit
    data_test_domain['CREDIT_TERM'] = data_test_domain['AMT_ANNUITY'] / data_test_domain['AMT_CREDIT']

    # Pourcentage des jours d'emploi par rapport à l'âge en jours
    data_test_domain['DAYS_EMPLOYED_PERCENT'] = data_test_domain['DAYS_EMPLOYED'] / data_test_domain['DAYS_BIRTH']

    # Explication :
    # Ce code calcule plusieurs nouvelles caractéristiques (features) pour les jeux de données d'entraînement et de test.
    # Ces caractéristiques sont créées en effectuant des opérations mathématiques sur les colonnes existantes.
    # Elles peuvent être utiles pour mieux comprendre les relations entre les variables et améliorer la performance des modèles de machine learning.
    # Par exemple, le pourcentage du crédit par rapport au revenu total peut donner des informations sur la capacité de remboursement d'un emprunteur.
    # De même, le pourcentage de l'annuité par rapport au revenu total peut aider à évaluer si un client peut gérer le paiement mensuel d'un crédit.
    # Ces caractéristiques nouvellement créées sont souvent appelées "ingénierie des caractéristiques" et font partie du processus d'exploration de données.

    return data_train_domain, data_test_domain

    #  
def load_and_preprocess_data(data_train, data_test):
    # Charger vos DataFrames train et test
    # data_train = pd.read_csv('../data/application_train.csv')
    # data_test = pd.read_csv('../data/application_test.csv')
    
    # Effectuer l'ingénierie des caractéristiques sur les DataFrames chargés
    df_train, df_test = features_engineering(data_train, data_test)
    
    return df_train, df_test

def train_logistic_regression(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.85, test_size=0.15, stratify=Y, random_state=123, shuffle=True)
    
    imputer = SimpleImputer(strategy='mean')  # Vous pouvez choisir une autre stratégie si nécessaire
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train, Y_train)

    return log_reg, X_test, X_train, Y_train, Y_test

def explain_sample_with_shap(log_reg, log_reg_explainer, X_test, sample_idx):
    shap_vals = log_reg_explainer.shap_values(X_test[sample_idx])

    if isinstance(log_reg, LogisticRegression):
        val1 = log_reg_explainer.expected_value + shap_vals[0]
    else:
        # Gérer le cas où votre modèle a plus de classes (plus de valeurs SHAP)
        # Vous devrez ajuster cela en fonction de la structure de votre modèle
        val1 = log_reg_explainer.expected_value[0] + shap_vals[0].sum()

    return val1, shap_vals
def generate_summary_plot(log_reg_explainer, X_test):
    shap.summary_plot(log_reg_explainer.shap_values(X_test),
                      feature_names=df_train.columns)

# Personnalisation de la barre latérale
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 300px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Personnalisation du contenu principal
st.markdown(
    """
    <style>
    .main {
        width: 1600px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# st.markdown(
#     """
#     <style>
#     .main {
#         width: 100% !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
import streamlit as st

st.markdown(
    """
    <style>
    .markdown-text-container {
        width: 800px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Votre contenu Streamlit ici

# # Personnalisation du contenu principal
# st.markdown(
#     """
#     <style>
#     .bloc-container {
#         width: 1400px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
# # Personnalisation du contenu principal
# st.markdown(
#     """
#     <style>
#     .element-container {
#         width: 1000px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Personnalisation du st.dataframe
# st.markdown(
#     """
#     <style>
#     .dataframe {
#         width: 500%; /* Largeur du tableau (peut être un pourcentage) */
#         height: 1000px; /* Hauteur du tableau */
#     }

#     .dataframe td, .dataframe th {
#         text-align: center; /* Alignement du texte dans les cellules */
#         padding: 6px; /* Marge intérieure des cellules */
#     }

#     .dataframe th {
#         background-color: #f2f2f2; /* Couleur de fond de l'en-tête de colonne */
#     }

#     .dataframe tr:nth-child(even) {
#         background-color: #f2f2f2; /* Couleur de fond des lignes paires */
#     }

#     .dataframe tr:hover {
#         background-color: #e3e3e3; /* Couleur de fond au survol des lignes */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
# Personnalisation de la taille, de la largeur et d'autres propriétés du st.dataframe
st.markdown(
    """
    <style>
    .dataframe {
        width: 120px  /* Largeur du tableau */
        height: 400px !important; /* Hauteur du tableau */
    }
    .dataframe td {
        text-align: center; /* Alignement du texte dans les cellules */
    }
    .dataframe th {
        background-color: #3498db !important; /* Couleur de fond de l'en-tête des colonnes */
        color: blue !important; /* Couleur du texte de l'en-tête des colonnes */
        text-align: center; /* Alignement du texte dans l'en-tête des colonnes */
    }
    .dataframe tr:nth-child(even) {
        background-color: #f2f2f2; /* Couleur de fond des lignes paires */
    }
    .dataframe tr:hover {
        background-color: #ffec99 !important; /* Couleur de fond au survol de la souris */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# AFFICHAGE DASHBOARD
init = st.markdown("*Initialisation de l'application en cours...*")
# On charge les données et on initialise l'application
data_train, data_test, data_train_prepared, data_test_prepared = load_data()
id_client = data_test["SK_ID_CURR"].values
clf_xgb = load_xgboost(data_train, data_test)

init = st.markdown("*Initialisation de l'application terminée...*")

# SIDEBAR
# Affichage du titre et du sous-titre
st.title("Implémenter un modèle de scoring")
st.markdown("<i>API répondant aux besoins du projet 7 pour le parcours Data Scientist OpenClassRoom</i>", unsafe_allow_html=True)

# Texte de présentation
st.sidebar.header("**PRET A DEPENSER**")

st.sidebar.subheader("Sélection ID_client")

# Chargement de la selectbox
chk_id = st.sidebar.selectbox("ID Client", id_client)

# Affichage d'informations dans la sidebar
st.sidebar.subheader("Informations générales")

# Chargement du logo
# Lors du déploiement sur Azure, l'affichage de l'image mettait le code en erreur.
# Car l'application n'arrivait pas à trouver le fichier.
# J'ai donc enlever cette partie pour le déploiement et l'ai remplacé par du texte.
# logo = load_logo()
# st.sidebar.image(logo, width=200)

# Chargement des infos générales
nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data_train)

# Affichage des infos dans la sidebar
# Nombre de crédits existants
st.sidebar.markdown("<u>Nombre crédits existants dans la base :</u>", unsafe_allow_html=True)
st.sidebar.text(nb_credits)

# Graphique camembert
st.sidebar.markdown("<u>Différence solvabilité / non solvabilité</u>", unsafe_allow_html=True)

plt.pie(targets, explode=[0, 0.1], labels=["Solvable", "Non solvable"], autopct='%1.1f%%', shadow=True, startangle=90)
st.sidebar.pyplot()

# Revenus moyens
st.sidebar.markdown("<u>Revenus moyens $(USD) :</u>", unsafe_allow_html=True)
st.sidebar.text(rev_moy)

# Montant crédits moyen
st.sidebar.markdown("<u>Montant crédits moyen $(USD) :</u>", unsafe_allow_html=True)
st.sidebar.text(credits_moy)

# PAGE PRINCIPALE
# Affichage de l'ID client sélectionné
st.write("Vous avez sélectionné le client :", chk_id)

# Affichage état civil
st.header("**Informations client**")
# infos_client = identite_client(data_test, chk_id)
if st.checkbox("Afficher les informations du client?"):

    infos_client = identite_client(data_test, chk_id)
    print(infos_client)
    st.write("Statut famille :**", infos_client["NAME_FAMILY_STATUS"].values[0], "**")
    st.write("Nombre d'enfant(s) :**", infos_client["CNT_CHILDREN"].values[0], "**")
    st.write("Age client :", int(infos_client["DAYS_BIRTH"] / -365), "ans.")

    data_age = load_age_population(data_train)
    # Set the style of plots
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(9, 9))
    # Plot the distribution of ages in years
    plt.hist(data_age, edgecolor='k', bins=25)
    plt.axvline(int(infos_client["DAYS_BIRTH"].values / -365), color="red", linestyle=":")
    plt.title('Age of Client')
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    st.pyplot()

    st.subheader("*Revenus*")
    # st.write("Total revenus client :", infos_client["revenus"], "$")
    st.write("Total revenus client :", infos_client["AMT_INCOME_TOTAL"].values[0], "$")

    data_revenus = load_revenus_population(data_train)
    # Set the style of plots
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(9, 9))
    # Plot the distribution of revenus
    plt.hist(data_revenus, edgecolor='k')
    plt.axvline(infos_client["AMT_INCOME_TOTAL"].values[0], color="red", linestyle=":")
    plt.title('Revenus du Client')
    plt.xlabel('Revenus ($ USD)')
    plt.ylabel('Count')
    st.pyplot()

    # st.write("Montant du crédit :", infos_client["montant_credit"], "$")
    # st.write("Annuités crédit :", infos_client["annuites"], "$")
    # st.write("Montant du bien pour le crédit :", infos_client["montant_bien"], "$")
    st.write("Montant du crédit :", infos_client["AMT_CREDIT"].values[0], "$")
    st.write("Annuités crédit :", infos_client["AMT_ANNUITY"].values[0], "$")
    st.write("Montant du bien pour le crédit :", infos_client["AMT_GOODS_PRICE"].values[0], "$")
else:
    st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)

# Affichage solvabilité client
st.header("**Analyse dossier client**")

st.markdown("<u>Probabilité de risque de faillite du client :</u>", unsafe_allow_html=True)
prediction = load_prediction(data_test, data_test_prepared, chk_id, clf_xgb)
st.write(round(prediction * 100, 2), "%")

st.markdown("<u>Données client :</u>", unsafe_allow_html=True)
# st.write("chk_id :", chk_id)
# st.write("data_test :", data_test)
infos_client_id = infos_client_id(data_test, chk_id)
# print(infos_client)
# st.write(pd.DataFrame(infos_client_id))
# Affichage du DataFrame avec plus d'espace
# st.dataframe(infos_client_id, width=800, height=800)  # Ajustez width et height selon vos besoins
# Définir les options de pandas pour afficher toutes les lignes et colonnes
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Maintenant, lorsque vous affichez un DataFrame, il n'y aura pas de limite
st.dataframe(infos_client_id)
# st.write(infos_client_id, unsafe_allow_html=True)
# Affichage des dossiers similaires
chk_voisins = st.checkbox("Afficher dossiers similaires?")

if chk_voisins:
    knn = load_knn(data_train_prepared)
    st.markdown("<u>Liste des 10 dossiers les plus proches de ce client :</u>", unsafe_allow_html=True)
    st.dataframe(load_voisins(data_train, data_test, data_test_prepared, chk_id, knn))
    st.markdown("<i>Target 1 = Client en faillite</i>", unsafe_allow_html=True)
else:
    st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)

# chk_shap = st.checkbox("Afficher les explications SHAP?")
# if chk_shap:
#     prediction, shap_values = load_prediction_with_shap(data_test, data_test_prepared, chk_id, clf_xgb)
#     st.markdown(f"<u>Probabilité de risque de faillite du client :</u> {round(prediction * 100, 2)}%", unsafe_allow_html=True)

#     st.markdown("<u>SHAP Values :</u>", unsafe_allow_html=True)
#     shap.summary_plot(shap_values, data_client, plot_type="bar")

#     st.markdown("<i>Les valeurs SHAP montrent l'impact de chaque feature sur la prédiction. Les features en rouge augmentent la probabilité de faillite, tandis que les features en bleu la réduisent.</i>", unsafe_allow_html=True)
# else:
#     st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.neighbors import NearestNeighbors
# import xgboost as xgb
# from joblib import load
# import shap
# import matplotlib.pyplot as plt

# @st.cache_data
# def load_knn(df):
#     print("Entrainement knn en cours...")
#     knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(df)
#     return knn

# # ... (votre code existant pour le chargement de données, le modèle XGBoost, etc.)

# @st.cache_data
# def load_knn(df_train):

#     knn = entrainement_knn(df_train)
#     print("Training knn done")

#     return knn

st.header("SHAP Explanations")

# chk_shap = st.checkbox("Afficher les explications SHAP pour k-NN?")
# if chk_shap:
#     # Affichage de l'ID client sélectionné
#     st.write("Vous avez sélectionné le client chk_id :", chk_id)
#     # st.dataframe(data_train_prepared)
#     # Chargez le modèle k-NN et les données (vous devez adapter cette partie en fonction de votre code)
#     knn_model = load_knn(data_train_prepared)
#     X = data_train_prepared  # Remplacez par vos données d'entraînement
#     st.dataframe(X)
#     feature_names = X.columns  # Remplacez par les noms de vos fonctionnalités
#     # Remplacez `chk_id` par l'ID du client que vous voulez expliquer
#     # Remarque : Assurez-vous d'adapter ces parties pour utiliser les données du client approprié
#     sample_idx = chk_id  # L'indice du client à expliquer
    
#     sample_data = data_train_prepared.iloc[sample_idx].values.reshape(1, -1)

#     # Utilisez SHAP pour expliquer la prédiction k-NN
#     explainer = shap.KernelExplainer(knn_model.kneighbors, X)  # Assurez-vous d'adapter cette partie en fonction de votre modèle k-NN
#     shap_values = explainer.shap_values(sample_data)

#     # Affichez les valeurs SHAP
#     st.markdown("<u>SHAP Values :</u>", unsafe_allow_html=True)
#     shap.initjs()  # Cette ligne est nécessaire pour afficher les graphiques SHAP dans Streamlit
#     shap.summary_plot(shap_values, X, feature_names=feature_names)

#     # Vous pouvez également afficher des graphiques SHAP individuels
#     st.markdown("<u>Graphique SHAP individuel :</u>", unsafe_allow_html=True)
#     shap.initjs()  # Cette ligne est nécessaire pour afficher les graphiques SHAP dans Streamlit
#     shap.force_plot(explainer.expected_value, shap_values, sample_data, feature_names=feature_names)
# # else:
# #     st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)
chk_shap = st.checkbox("Afficher les explications SHAP ")
# if chk_shap:
#     # Initialisation de SHAP
#     shap.initjs()
#     # Chargez le modèle k-NN et les données
#     df_train, df_test = load_and_preprocess_data(data_train, data_test)

#     X = df_train
#     Y = df_train['TARGET']
#     log_reg, X_test, X_train, Y_train, Y_test = train_logistic_regression(X, Y)
#     log_reg_explainer = shap.LinearExplainer(log_reg, X_train)

#     sample_idx = 0
#     val1, shap_vals = explain_sample_with_shap(log_reg, log_reg_explainer, X_test, sample_idx)
#     shap.summary_plot(log_reg_explainer.shap_values(X_test), feature_names=df_train.columns)
#     # generate_summary_plot(log_reg_explainer, X_test)
# if chk_shap:
#     # Initialisation de SHAP
#     shap.initjs()
#     # Chargez vos données et entraînez le modèle
#     data_train, data_test = load_and_preprocess_data(data_train, data_test)
#     X = data_train
#     Y = data_train['TARGET']
#     log_reg, X_test, X_train, Y_train, Y_test = train_logistic_regression(X, Y)
#     log_reg_explainer = shap.LinearExplainer(log_reg, X_train)
#     num_samples=1
#     # sample_idx = 0
#     # val1, shap_vals = explain_sample_with_shap(log_reg, log_reg_explainer, X_test, sample_idx)
#     # Générer les valeurs SHAP pour un échantillon
#     shap_values = log_reg_explainer.shap_values(X_test[num_samples])
#     # Assurez-vous que feature_names correspond à vos données
#     # feature_names = data_train.columns  # Remplacez ceci par les noms de vos fonctionnalités

#     # # Générer le graphique SHAP pour l'échantillon actuel
#     # shap.bar_plot(shap_values[0], feature_names=feature_names)
    
#     st.write("X_test :", X_test)
#     # st.write("Valeur SHAP :", val1)
#     st.write("Détails shap_values(X_test) :", shap_values)
#     # shap.summary_plot(log_reg_explainer.shap_values, feature_names=data_train.columns)
#     expected_value = [log_reg_explainer.expected_value.tolist()]  # Convertit la valeur attendue en liste
#     shap_values = [log_reg_explainer.shap_values(X_test)]  # Convertit les valeurs SHAP en liste
#     feature_names = data_train.columns.tolist()  # Convertit les noms de colonnes en liste

#     shap.multioutput_decision_plot(expected_value, shap_values, row_index=0, feature_names=feature_names)
if chk_shap:
    # Initialisation de SHAP
    shap.initjs()
    # Chargez vos données et entraînez le modèle
    data_train, data_test = load_and_preprocess_data(data_train, data_test)
    X = data_train
    Y = data_train['TARGET']
    log_reg, X_test, X_train, Y_train, Y_test = train_logistic_regression(X, Y)
    log_reg_explainer = shap.LinearExplainer(log_reg, X_train)

    num_samples = 10
    sample_indices = range(num_samples)
    shap_values = log_reg_explainer.shap_values(X_test[sample_indices])
    # st.write("Détails shap_values(X_test) :", shap_values)
    # Assurez-vous que feature_names correspond à vos données
    feature_names = data_train.columns.tolist()
    # st.write("feature_names :", feature_names)
    # Générer le graphique SHAP résumé pour les échantillons
    # shap.summary_plot(shap_values, feature_names=feature_names)
    # shap.summary_plot(log_reg_explainer.shap_values(X_test),feature_names=feature_names)
     # Générer le graphique SHAP en barres
    st.write("Graphique SHAP en barres :")
    shap.bar_plot(log_reg_explainer.shap_values(X_test[1]), feature_names=data_train.columns.tolist())
    
    # Générer un graphique SHAP de type "force plot"
    st.write("Graphique SHAP de type Force Plot :")
    shap.force_plot(log_reg_explainer.expected_value, log_reg_explainer.shap_values(X_test[1]), feature_names=data_train.columns.tolist())
    
    # Générer un graphique SHAP de dépendance pour une seule caractéristique
    # feature_of_interest = 'Feature_Name'  # Remplacez par le nom de la caractéristique
    # st.write(f"Graphique SHAP de dépendance pour la caractéristique '{feature_names}' :")
    # shap.dependence_plot(feature_names, log_reg_explainer.shap_values(X_test), X_test)
    
    st.write("FIN")

