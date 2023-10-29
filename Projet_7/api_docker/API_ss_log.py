from joblib import load

from flask import Flask, jsonify, request, jsonify, render_template
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE

from imblearn.under_sampling  import RandomUnderSampler

import xgboost as xgb

import shap
# import ipython

# Create an instance of the Flask class that is the WSGI application.
# The first argument is the name of the application module or package,
# typically __name__ when using a single module.
# app = Flask(__name__)
# from flask_cors import CORS

app = Flask(__name__)
# CORS(app)  # Activer CORS pour gérer les demandes Cross-Origin (si nécessaire)

# Flask route decorators map / and /hello to the hello function.
# To add other resources, create functions that generate the page contents
# and add decorators to define the appropriate resource locators for them.

# On charge les données
data_train = pd.read_csv("application_train.csv")
data_test = pd.read_csv("application_test.csv")

# On crée deux variables en attente qui deviendront
# des variables globales après l'initialisation de l'API.
# Ces variables sont utilisées dans plusieurs fonctions de l'API.
train = None
test = None
model = None

# On crée la liste des ID clients qui nous servira dans l'API
id_client = data_test["SK_ID_CURR"][:50].values
id_client = pd.DataFrame(id_client)

# routes
# Entraînement du modèle
@app.route("/init_model", methods=["GET"])
def init_model():
    
    # On prépare les données
    df_train, df_test = features_engineering(data_train, data_test)

    print("Features engineering done")
    # On fait le préprocessing des données
    df_train, df_test = preprocesseur(df_train, df_test)

    # On transforme le dataset de test préparé en variabe
    # globale, car il est utilisé dans la fonction predict
    global train
    train = df_train.copy()

    global test
    test = df_test.copy()

    print("Preprocessing done")
    # On fait un resampling des données d'entraînement
    X, y = data_resampler(df_train, data_train)
    print("Resampling done")

    # Équilibrage des données d'entraînement avec SMOTE
    X, y = data_resampler_SMOTE(df_train, data_train)
    print("Resampling SMOTE done")    

    # On entraîne le modèle et on le transforme en
    # variable globale pour la fonction predict
    global clf_xgb
    clf_xgb = entrainement_XGBoost(X, y)
    print("Training xgboost done")

    global knn
    knn = entrainement_knn(df_train)
    print("Training knn done")

    return jsonify(["Initialisation terminée."])
# @app.route("/init_model", methods=["GET"])
# def init_model():
#     with mlflow.start_run():
#         # On prépare les données
#         df_train, df_test = features_engineering(data_train, data_test)
#         print("Features engineering done")

#         # On fait le préprocessing des données
#         df_train, df_test = preprocesseur(df_train, df_test)
#         global train
#         train = df_train.copy()
#         global test
#         test = df_test.copy()
#         print("Preprocessing done")

#         # On fait un resampling des données d'entraînement
#         X, y = data_resampler(df_train, data_train)
#         print("Resampling done")

#         # Entraînement du modèle
#         global clf_xgb
#         clf_xgb = entrainement_XGBoost(X, y)
#         print("Training xgboost done")

#         # Enregistrement du modèle dans MLflow
#         mlflow.xgboost.log_model(clf_xgb, "xgboost_model")

#         # Enregistrement de métriques
#         mlflow.log_params({
#             "n_estimators": clf_xgb.n_estimators,
#             "learning_rate": clf_xgb.learning_rate,
#             "max_depth": clf_xgb.max_depth
#         })
        
#         return jsonify(["Initialisation terminée."])

# Chargement des données pour la selection de l'ID client
@app.route("/load_data", methods=["GET"])
def load_data():
    
    return id_client.to_json(orient='values')

# Chargement d'informations générales
@app.route("/infos_gen", methods=["GET"])
def infos_gen():

    lst_infos = [data_train.shape[0],
                 round(data_train["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data_train["AMT_CREDIT"].mean(), 2)]

    return jsonify(lst_infos)

# Chargement des données pour le graphique
# dans la sidebar
@app.route("/disparite_target", methods=["GET"])
def disparite_target():

    df_target = data_train["TARGET"].value_counts()

    return df_target.to_json(orient='values')

# Chargement d'informations générales sur le client
@app.route("/infos_client", methods=["GET"])
def infos_client():

    id = request.args.get("id_client")

    data_client = data_test[data_test["SK_ID_CURR"] == int(id)]

    print(data_client)
    dict_infos = {
       "status_famille" : data_client["NAME_FAMILY_STATUS"].item(),
       "nb_enfant" : data_client["CNT_CHILDREN"].item(),
       "age" : int(data_client["DAYS_BIRTH"].values / -365),
       "revenus" : data_client["AMT_INCOME_TOTAL"].item(),
       "montant_credit" : data_client["AMT_CREDIT"].item(),
       "annuites" : data_client["AMT_ANNUITY"].item(),
       "montant_bien" : data_client["AMT_GOODS_PRICE"].item()
       }
    
    response = json.loads(data_client.to_json(orient='index'))

    return response

# Calcul des ages de la population pour le graphique
# situant l'age du client
@app.route("/load_age_population", methods=["GET"])
def load_age_population():
    
    df_age = round((data_train["DAYS_BIRTH"] / -365), 2)
    return df_age.to_json(orient='values')

# Segmentation des revenus de la population pour le graphique
# situant l'age du client
@app.route("/load_revenus_population", methods=["GET"])
def load_revenus_population():
    
    # On supprime les outliers qui faussent le graphique de sortie
    # Cette opération supprime un peu moins de 300 lignes sur une
    # population > 300000...
    df_revenus = data_train[data_train["AMT_INCOME_TOTAL"] < 700000]
    
    df_revenus["tranches_revenus"] = pd.cut(df_revenus["AMT_INCOME_TOTAL"], bins=20)
    df_revenus = df_revenus[["AMT_INCOME_TOTAL", "tranches_revenus"]]
    df_revenus.sort_values(by="AMT_INCOME_TOTAL", inplace=True)

    print(df_revenus)
    
    df_revenus = df_revenus["AMT_INCOME_TOTAL"]

    return df_revenus.to_json(orient='values')

@app.route("/predict", methods=["GET"])
def predict():
    
    id = request.args.get("id_client")

    print("Analyse data_test :")
    print(data_test.shape)
    print(data_test[data_test["SK_ID_CURR"] == int(id)])
     
    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    print(index[0])
   
    data_client = test[index]

    print(data_client)

    prediction = clf_xgb.predict_proba(data_client)

    prediction = prediction[0].tolist()

    print(prediction)

    return jsonify(prediction)
# Créez un explainer SHAP pour le modèle
# explainer = shap.Explainer(clf_xgb, train)

# @app.route("/predict_explanation", methods=["GET"])
# def predict_explanation():
#     explainer = shap.Explainer(clf_xgb, train)
#     id = request.args.get("id_client")
#     index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values
#     data_client = test[index]
    
#     # Obtenez les valeurs SHAP pour la prédiction
#     shap_values = explainer(data_client)

#     # Transformez les valeurs SHAP en DataFrame
#     shap_df = pd.DataFrame(shap_values.values, columns=data_test.columns)

#     # Créez des graphiques SHAP
#     shap.summary_plot(shap_values, data_client, show=False)
#     plt.savefig('shap_summary_plot.png')

#     # Obtenez un résumé des valeurs SHAP
#     summary = shap.summary_plot(shap_values, data_client, show=False, plot_type='bar', max_display=10)
#     plt.savefig('shap_summary.png')

#     return jsonify({
#         "shap_values": shap_df.to_dict(orient="split"),
#         "shap_summary_plot": "shap_summary_plot.png",
#         "shap_summary": "shap_summary.png"
#     })
# @app.route("/predict_explanation", methods=["GET"])
# def predict_explanation():
#     explainer = shap.Explainer(clf_xgb, train)
#     id = request.args.get("id_client")
#     index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values
#     data_client = test[index]
    
#     # Obtenez les valeurs SHAP pour la prédiction
#     shap_values = explainer(data_client)

#     return shap_values , data_client , data_test
# @app.route("/predict_explanation", methods=["GET"])
# def predict_explanation():

#     explainer = shap.Explainer(clf_xgb, train)
#     id = request.args.get("id_client")
#     index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values
#     data_client = test[index]
    
#     # Obtenez les valeurs SHAP pour la prédiction
#     shap_values = explainer(data_client)

#     # Transformez les valeurs SHAP en DataFrame
#     shap_df = pd.DataFrame(shap_values.values, columns=data_test.columns)

#     return shap_df.to_dict(orient="split")

# @app.route("/predict_explanation", methods=["GET"])
# def predict_explanation():

#     id = request.args.get("id_client")
#     index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values
#     data_client = test[index]
    
#     # Obtenez les valeurs SHAP pour la prédiction en utilisant shap.KernelExplainer
#     shap_values = get_shap_values(data_client)

#     # Transformez les valeurs SHAP en DataFrame
#     shap_df = pd.DataFrame(shap_values, columns=data_test.columns)

#     return shap_df.to_dict(orient="split")

# def get_shap_values(data_client):
    
#     explainer = shap.KernelExplainer(clf_xgb.predict_proba, train)
    
#     # Calcul des valeurs SHAP pour le client
#     shap_values = explainer.shap_values(data_client)

#     return shap_values

def train_logistic_regression(X, Y):
    st.write("debut train_logistic_regression dans api")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.85, test_size=0.15, stratify=Y, random_state=123, shuffle=True)
    
    imputer = SimpleImputer(strategy='mean')  # Vous pouvez choisir une autre stratégie si nécessaire
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train, Y_train)
    st.write("Fin train_logistic_regression dans api")
    return log_reg, X_test, X_train, Y_train, Y_test      
@app.route("/grap_shap", methods=["GET"])                    
def grap_shap():
    # st.write("debut grap_shap dans api")
    # warnings.filterwarnings("ignore")
    # print("Debut grap_shap ")
    # df_train, df_test = load_and_preprocess_data()
    df_train.head(1)
#     X, Y = df_train, df_train.target
    X = df_train
    Y = df_train['TARGET']
    # print("Total Data Size : ", X.shape, Y.shape)

    log_reg, X_test, X_train, Y_train, Y_test = train_logistic_regression(X, Y)
    # print("Test  Accuracy : ", log_reg.score(X_test, Y_test))
    # print("Train Accuracy : ", log_reg.score(X_train, Y_train))
 
    log_reg_explainer = shap.LinearExplainer(log_reg, X_train)

    sample_idx = 0
#     val1, val2, val3, shap_vals = explain_sample_with_shap(log_reg, log_reg_explainer, X_test, sample_idx)
    val1, shap_vals = explain_sample_with_shap(log_reg, log_reg_explainer, X_test, sample_idx)

#     print("Expected/Base Values : ", log_reg_explainer.expected_value)
#     print("Shap Values for Sample %d : " % sample_idx, shap_vals)
#     print("Prediction From Model                            : ", \
#           wine.target_names[log_reg.predict(X_test[sample_idx].reshape(1, -1))[0]])
#     print("Prediction From Adding SHAP Values to Base Value : ", wine.target_names[np.argmax([val1, val2, val3])])

#     # Générer les graphiques
#     generate_shap_summary_plot(log_reg_explainer, X_test, num_samples=1)
#     generate_shap_bar_plots(log_reg_explainer, X_test, num_samples=1)
#     generate_waterfall_plots(log_reg_explainer, X_test)
    # generate_decision_plot(log_reg_explainer, X_test) # fonctionne 
#     generate_dependence_plots(log_reg_explainer, X_test)
#     generate_embedding_plots(log_reg_explainer, X_test)
#     generate_force_plots(log_reg_explainer, X_test)
    # generate_summary_plot(log_reg_explainer, X_test) # fonctionne
#     generate_partial_dependence_plots(log_reg, X_test, wine)
    # st.write("Fin grap_shap dans api")
    return log_reg_explainer , X_test
@app.route("/load_voisins", methods=["GET"])
def load_voisins():
    
    id = request.args.get("id_client")

    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    data_client = test[index]
    
    distances, indices = knn.kneighbors(data_client)

    print("indices")
    print(indices)
    print("distances")
    print(distances)

    df_voisins = data_train.iloc[indices[0], :]
    
    response = json.loads(df_voisins.to_json(orient='index'))

    return response


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

def preprocesseur(df_train, df_test):
    
    # Cette fonction permet d'imputer les valeurs manquantes dans
    # chaque dataset et aussi d'appliquer un MinMaxScaler

    # Drop the target from the training data
    if "TARGET" in df_train:
        train = df_train.drop(columns = ["TARGET"])
    else:
        train = df_train.copy()
        
    # Feature names
    features = list(train.columns)


    # Median imputation of missing values
    imputer = SimpleImputer(strategy = 'median')

    # Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range = (0, 1))

    # Replace the boolean column by numerics values 
    train["DAYS_EMPLOYED_ANOM"] = train["DAYS_EMPLOYED_ANOM"].astype("int")

    # Fit on the training data
    imputer.fit(train)

    # Transform both training and testing data
    train = imputer.transform(train)
    test = imputer.transform(df_test)

    # Repeat with the scaler
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    
    return train, test

def data_resampler(df_train, target):

    rsp = RandomUnderSampler()
    X_rsp, y_rsp = rsp.fit_resample(df_train, target["TARGET"])

    return X_rsp, y_rsp

# Fonction pour l'équilibrage des données avec SMOTE
def data_resampler_SMOTE(df_train, target):
    smote = SMOTE(sampling_strategy='auto', random_state=0)
    X_smote, y_smote = smote.fit_resample(df_train, target["TARGET"])
    return X_smote, y_smote

def entrainement_XGBoost(X, y):

    # Configuration de la meilleure itération trouvée par le RandomizeSearchCV
    # Optimized n_estimator=1144
    clf_xgb = xgb.XGBClassifier(booster='gbtree',
                                colsample_bytree=0.6784538670198459,              
                                eval_metric='auc',
                                learning_rate=0.10310087264740633,
                                max_depth=6, 
                                min_child_weight=2,       
                                n_estimators=150,         
                                objective='binary:logistic', 
                                random_state=0,             
                                subsample=0.4915549740714592,
                                n_jobs = -1
                                )

    clf_xgb.fit(X, y)

    return clf_xgb

def entrainement_knn(df):

    print("En cours...")
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(df)

    return knn 

# @app.route("/mlflow_version", methods=["GET"])
def mlflow_version():
    mlflow_version = mlflow.__version__
    return f"MLflow Version: {mlflow_version}"

if __name__ == "__main__":
    # app.run(host="localhost", port="5000", debug=True)
    app.run(host="0.0.0.0", port="5000", debug=True)    
