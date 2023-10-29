# Importation des bibliothèques
import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import shap
# import ipython
from PIL import Image
from flask import Flask

# from "C:\Users\Zbook\OpenClassRoom\ProjetsGitsOCR\Projet_7\api_docker\api" import get_shap_values, predict_explanation


# Définition de l'URL de l'API
# URL_API = "http://localhost:5000/" # Utilisation en local
URL_API = "http://projet7API:5000/"  # Utilisation en production

def main():

    # Initialisation de l'application
    init = st.markdown("*Initialisation de l'application en cours...*")
    init = st.markdown(init_api())
    shap.initjs()
    # st.write("Version mlflow: ", mlflow_version())
    # Affichage du titre et du sous-titre
    st.title("Implémenter un modèle de scoring")
    st.markdown("<i>API répondant aux besoins du projet 7 pour le parcours Data Scientist OpenClassRoom</i>", unsafe_allow_html=True)

    # Affichage d'informations dans la sidebar
    st.sidebar.subheader("Informations générales")
    
    # Chargement du logo
    logo = load_logo()
    st.sidebar.image(logo, width=200)

    # Chargement de la selectbox
    lst_id = load_selectbox()
    global id_client
    id_client = st.sidebar.selectbox("ID Client", lst_id)

    # Chargement des infos générales
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen()

    # Affichage des infos dans la sidebar
    st.sidebar.markdown("<u>Nombre crédits existants dans la base :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    # # Graphique camembert
    # st.sidebar.markdown("<u>Différence solvabilité / non solvabilité</u>", unsafe_allow_html=True)
    # plt.pie(targets, explode=[0, 0.1], labels=["Solvable", "Non solvable"], autopct='%1.1f%%', shadow=True, startangle=90)
    # st.sidebar.pyplot()
    # Graphique camembert
    st.sidebar.markdown("<u>Différence solvabilité / non solvabilité</u>", unsafe_allow_html=True)
    # Désactivez l'avertissement lié à Matplotlib
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, ax = plt.subplots()
    ax.pie(targets, explode=[0, 0.1], labels=["Solvable", "Non solvable"], autopct='%1.1f%%', shadow=True, startangle=90)
    st.sidebar.pyplot(fig)


    # Revenus moyens
    st.sidebar.markdown("<u>Revenus moyens $(USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    # Montant crédits moyen
    st.sidebar.markdown("<u>Montant crédits moyen $(USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)

    # Affichage de l'ID client sélectionné
    st.write("Vous avez sélectionné le client :", id_client)

    # Affichage état civil
    st.header("**Informations client**")
    infos_client = identite_client()
    if st.checkbox("Afficher les informations du client?"):
        infos_client = identite_client()
        st.write("Statut famille :**", infos_client["NAME_FAMILY_STATUS"][0], "**")
        st.write("Nombre d'enfant(s) :**", infos_client["CNT_CHILDREN"][0], "**")
        st.write("Age client :", int(infos_client["DAYS_BIRTH"].values / -365), "ans.")

        data_age = load_age_population()
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(9, 9))
        plt.hist(data_age, edgecolor='k', bins=25)
        plt.axvline(int(infos_client["DAYS_BIRTH"].values / -365), color="red", linestyle=":")
        plt.title('Age of Client')
        plt.xlabel('Age (years)')
        plt.ylabel('Count')
        # Avant d'afficher les graphiques, désactivez l'avertissement
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.pyplot()

        # st.subheader("*Revenus*")
        # st.write("Total revenus client :", infos_client["AMT_INCOME_TOTAL"][0], "$")

        # data_revenus = load_revenus_population()
        # plt.style.use('fivethirtyeight')
        # plt.figure(figsize=(9, 9))
        # plt.hist(data_revenus, edgecolor='k')
        # plt.axvline(infos_client["AMT_INCOME_TOTAL"][0], color="red", linestyle=":")
        # plt.title('Revenus du Client')
        # plt.xlabel('Revenus ($ USD)')
        # plt.ylabel('Count')
        # # Avant d'afficher les graphiques, désactivez l'avertissement
        # st.set_option('deprecation.showPyplotGlobalUse', False)
    
        # st.pyplot()
        st.subheader("*Revenus*")
        st.write("Total revenus client :", infos_client["AMT_INCOME_TOTAL"][0], "$")

        data_revenus = load_revenus_population()

        # Set the style of plots
        plt.style.use('fivethirtyeight')
        fig, ax = plt.subplots(figsize=(9, 9))

        # Plot the distribution of revenus
        ax.hist(data_revenus, edgecolor='k')
        ax.axvline(infos_client["AMT_INCOME_TOTAL"][0], color="red", linestyle=":")
        ax.set_title('Revenus du Client')
        ax.set_xlabel('Revenus ($ USD)')
        ax.set_ylabel('Count')

        # Avant d'afficher les graphiques, désactivez l'avertissement
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.pyplot(fig)

        # 

        st.write("Montant du crédit :", infos_client["AMT_CREDIT"][0], "$")
        st.write("Annuités crédit :", infos_client["AMT_ANNUITY"][0], "$")
        st.write("Montant du bien pour le crédit :", infos_client["AMT_GOODS_PRICE"][0], "$")
    else:
        st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)

    # Affichage solvabilité client
    st.header("**Analyse dossier client**")
    st.markdown("<u>Probabilité de risque de faillite du client :</u>", unsafe_allow_html=True)
    prediction = load_prediction(id_client)
    st.write(round(prediction * 100, 2), "%")
    st.markdown("<u>Données client :</u>", unsafe_allow_html=True)
    st.write(identite_client())
# 
    # Affichage des informations du client et explication avec SHAP
    # st.header("Informations client et explication SHAP")
    # shap.initjs()
    # if st.checkbox("Afficher les informations du client et explication SHAP?"):
    #     # Demandez les données SHAP à votre API
    #     shap_data = get_shap_data(id_client)
    #     st.write("SHAP 1 id_client", id_client)
    #     st.write("SHAP 2 shap_data", shap_data)
    #     # Générez les graphiques SHAP dans Streamlit
    #     if shap_data:
    #         st.write("Graphique SHAP Summary :")
    #         shap_summary_plot = generate_shap_summary_plot(shap_data)
    #         st.write("SHAP 3 shap_summary_plot", shap_summary_plot)    
    #         st.pyplot(shap_summary_plot)
    #         st.header("SHAP Values")

    #         # Afficher les valeurs SHAP pour une prédiction spécifique
    #         st.write("SHAP 4 Values for Prediction", id_client)
    #         shap_values = shap_data["shap_values"]
    #         st.write("SHAP shap_values", shap_values)
    #         expected_value = shap_data["expected_value"]
    #         st.write("SHAP expected_value", expected_value)            
    #         data_for_prediction = shap_data["data_for_prediction"]
    #         st.write("SHAP data_for_prediction", data_for_prediction)           

    #         shap.force_plot(expected_value, shap_values, data_for_prediction)
    # Affichage des dossiers similaires
    chk_voisins = st.checkbox("Afficher dossiers similaires?")

    if chk_voisins:
        similar_id = load_voisins()
        st.markdown("<u>Liste des 10 dossiers les plus proches de ce client :</u>", unsafe_allow_html=True)
        st.write(similar_id)
        st.markdown("<i>Target 1 = Client en faillite</i>", unsafe_allow_html=True)
    else:
        st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)

    if st.checkbox("Afficher les informations du client et explication SHAP?"):
        # shap_data = grap_shap()  # Appel de la fonction pour obtenir les valeurs SHAP
        # Appel à la fonction pour afficher les graphiques SHAP
        plot_shap_graphs()        
        st.write("SHAP 1 id_client", id_client)
        # st.write("SHAP 2 shap_data", shap_data)
        # Générez les graphiques SHAP dans Streamlit
        # if shap_data:
        #     st.write("Graphique SHAP Summary :")
        #     shap_summary_plot = generate_shap_summary_plot(shap_data)
        #     st.write("SHAP 3 shap_summary_plot", shap_summary_plot)    
        #     st.pyplot(shap_summary_plot)
        #     st.header("SHAP Values")

        #     # Afficher les valeurs SHAP pour une prédiction spécifique
        #     st.write("SHAP 4 Values for Prediction", id_client)
        #     shap_values = shap_data["shap_values"]
        #     st.write("SHAP shap_values", shap_values)
        #     expected_value = shap_data["expected_value"]
        #     st.write("SHAP expected_value", expected_value)            
        #     data_for_prediction = shap_data["data_for_prediction"]
        #     st.write("SHAP data_for_prediction", data_for_prediction)           

        #     shap.force_plot(expected_value, shap_values, data_for_prediction)


            # shap.summary_plot(shap_values, subsampled_test_data, feature_names=X_train.columns, max_display=10)

        # Utiliser shap.force_plot pour afficher un graphique SHAP interactif

        # shap.force_plot(explainer.expected_value, shap_values[id_client], data.iloc[id_client])
    # st.header("Informations client et explication SHAP")

    # if st.checkbox("Afficher les informations du client et explication SHAP?"):
    #     # Autres informations sur le client

    #     st.header("SHAP Values")

    #     # Afficher les valeurs SHAP pour une prédiction spécifique
    #     st.write("SHAP Values for Prediction", id_client)

    #     # Utiliser shap.force_plot pour afficher un graphique SHAP interactif
    #     shap.initjs()
        
    #     # Obtenez les données SHAP spécifiques pour le client sélectionné
    #     shap_data = get_shap_data(id_client)
        
    #     # Vérifiez si les données SHAP sont disponibles
    #     if shap_data:
    #         shap_values = shap_data["shap_values"]
    #         st.write("SHAP shap_values", shap_values)
    #         expected_value = shap_data["expected_value"]
    #         st.write("SHAP expected_value", expected_value)            
    #         data_for_prediction = shap_data["data_for_prediction"]
    #         st.write("SHAP data_for_prediction", data_for_prediction)           

    #         shap.force_plot(expected_value, shap_values, data_for_prediction)
    #     else:
    #         st.write("Données SHAP non disponibles pour ce client.")


        # infos_client = identite_client()
        # st.write("Statut famille :", infos_client["NAME_FAMILY_STATUS"][0])
        # st.write("Nombre d'enfants :", infos_client["CNT_CHILDREN"][0])
        # st.write("Âge client :", int(infos_client["DAYS_BIRTH"].values / -365), "ans")

        # # ... (autres informations)

        # # Affichage des graphiques SHAP
        # st.subheader("Explication SHAP des prédictions")
        # explanation_data , data_client , data_test = predict_explanation(id_client)
        # # Transformez les valeurs SHAP en DataFrame
        # shap_df = pd.DataFrame(explanation_data.values, columns=data_test.columns)

        # # Créez des graphiques SHAP
        # shap.summary_plot(explanation_data, data_client, show=False)

        # # Obtenez un résumé des valeurs SHAP
        # summary = shap.summary_plot(explanation_data, data_client, show=False, plot_type='bar', max_display=10)
    
        # shap_values = explanation_data["shap_values"]
        # shap_summary_plot = explanation_data["shap_summary_plot"]
        # shap_summary = explanation_data["shap_summary"]

        # # Affichez le résumé SHAP sous forme de graphique
        # st.image(shap_summary_plot)

        # # Affichez un résumé des valeurs SHAP sous forme de graphique
        # st.image(shap_summary)

        # ... (autres graphiques)

    # ... (autres parties du tableau de bord)
# 


# Fonction pour l'initialisation de l'API
@st.cache_data
def init_api():
    init_api = requests.get(URL_API + "init_model")
    init_api = init_api.json()
    return "Initialisation application terminée."

# Fonction pour charger le logo
@st.cache_data
def load_logo():
    logo = Image.open("logo.png")
    return logo

# Fonction pour charger la liste des ID clients
@st.cache_data
def load_selectbox():
    data_json = requests.get(URL_API + "load_data")
    data = data_json.json()
    lst_id = [i[0] for i in data]
    return lst_id

# Fonction pour charger les informations générales
@st.cache_data
def load_infos_gen():
    infos_gen = requests.get(URL_API + "infos_gen")
    infos_gen = infos_gen.json()
    nb_credits = infos_gen[0]
    rev_moy = infos_gen[1]
    credits_moy = infos_gen[2]
    targets = requests.get(URL_API + "disparite_target")
    targets = targets.json()
    return nb_credits, rev_moy, credits_moy, targets

# Fonction pour récupérer les informations du client sélectionné
@st.cache_data
def identite_client():
    infos_client = requests.get(URL_API + "infos_client", params={"id_client": id_client})
    infos_client = json.loads(infos_client.content.decode("utf-8"))
    infos_client = pd.DataFrame.from_dict(infos_client).T
    return infos_client

# Fonction pour charger l'âge de la population
@st.cache_data
def load_age_population():
    data_age_json = requests.get(URL_API + "load_age_population")
    data_age = data_age_json.json()
    return data_age

@st.cache_data  # 👈 Add the caching decorator
def load_revenus_population():    
    # Requête permettant de récupérer des tranches de revenus 
    # de la population pour le graphique situant le client
    data_revenus_json = requests.get(URL_API + "load_revenus_population")
    
    data_revenus = data_revenus_json.json()

    return data_revenus
  

# Fonction pour charger les load_prediction_explanation
# @st.cache_data
# def predict_explanation(id_client):
#     try:
#         explanation_data = requests.get(URL_API + "predict_explanation", params={"id_client": id_client})
#         explanation_data = explanation_data.json()
#         if explanation_data:
#             return explanation_data
#         else:
#             return {}  # Retourner un dictionnaire vide ou une autre valeur par défaut
#     except json.JSONDecodeError as e:
#         st.error(f"Erreur lors de la récupération des explications : {e}")
#         return {}
# @st.cache_data
# def predict_explanation(id_client):
#     try:
#         explanation_data = requests.get(URL_API + "predict_explanation", params={"id_client": id_client})
#         explanation_data = explanation_data.json()
        
#         if "shap_values" in explanation_data:
#             return explanation_data["shap_values"]
#         else:
#             st.error("Les données SHAP ne sont pas disponibles dans la réponse.")
#             return {}
#     except json.JSONDecodeError as e:
#         st.error(f"Erreur lors de la récupération des explications : {e}")
#         return {}
# @st.cache_data
# def get_shap_data(id_client):
#     try:
#         shap_data = requests.get(URL_API + "predict_explanation", params={"id_client": id_client})
#         shap_data = shap_data.json()
#         if shap_data:
#             return shap_data
#     except ValueError:
#         st.error("Erreur lors de la récupération des données SHAP.")
#     return None
def generate_decision_plot(log_reg_explainer, X_test, row_index=0, highlight=None):
    expected_value = [log_reg_explainer.expected_value.tolist()]  # Convertit la valeur attendue en liste
    shap_values = [log_reg_explainer.shap_values(X_test)]  # Convertit les valeurs SHAP en liste
    feature_names = df_train.columns.tolist()  # Convertit les noms de colonnes en liste

    shap.multioutput_decision_plot(expected_value, shap_values, row_index=row_index, feature_names=feature_names, highlight=highlight)
def generate_summary_plot(log_reg_explainer, X_test):
    shap.summary_plot(log_reg_explainer.shap_values(X_test),
                      feature_names=df_train.columns)
# @st.cache_data
# def grap_shap():
#     try:
#         st.write("debut grap_shap dans dashboard")
#         log_reg_explainer , X_test = requests.get(URL_API + "grap_shap") #, params={"id_client": id_client})
#         # st.write("grap_shap shap_data", shap_data)
#         st.write("grap_shap id_client", id_client)
#         generate_decision_plot(log_reg_explainer, X_test)    
#         generate_summary_plot(log_reg_explainer, X_test)
#         # if not shap_data.text:
#         #     st.error("Erreur lors de la récupération des données SHAP, shap_data.text est vide.")
#         #     return None  # Retournez None si la réponse est vide
#         # # shap_data = shap_data.json()
#         # return shap_data
#         st.write("Fin grap_shap dans dashboard")
#     except ValueError:
#         st.error("Erreur lors de la récupération des données SHAPdans la fonction grap_shap dans le dahsboard.")
#         return None


# Fonction pour afficher les graphiques SHAP
# @st.cache_data
# def plot_shap_graphs():
#     try:
#         response = requests.get(URL_API + "grap_shap") #, params={"id_client": id_client})
#         st.write(f"response : {response}")
#         if response.status_code == 200:
#             log_reg_explainer, X_test = response.json()
#             expected_value = [log_reg_explainer.expected_value.tolist()]
#             shap_values = [log_reg_explainer.shap_values(X_test)]
#             feature_names = X_test.columns.tolist()
#             st.write(f"feature_names : {feature_names}")
#             # Créez des graphiques SHAP
#             shap.multioutput_decision_plot(expected_value, shap_values, feature_names=feature_names)

#             shap.summary_plot(log_reg_explainer.shap_values(X_test), feature_names=X_test.columns)

#     except Exception as e:
#         st.error(f"Erreur lors de l'affichage des graphiques SHAP : {e}")
@st.cache_data
def plot_shap_graphs():
    try:
        # Faites une demande GET à votre API
        response = requests.get(URL_API + "grap_shap")  # Remplacez "grap_shap" par votre URL d'API
        st.write(f"response : {response}")

        if response.status_code == 200:
            log_reg_explainer, X_test = response.json()
            expected_value = log_reg_explainer.expected_value
            shap_values = log_reg_explainer.shap_values(X_test)
            feature_names = X_test.columns.tolist()
            st.write(f"feature_names : {feature_names}")
            
            # Créez des graphiques SHAP
            shap.multioutput_decision_plot(expected_value, shap_values, feature_names=feature_names)
            shap.summary_plot(shap_values, feature_names=feature_names)

    except Exception as e:
        st.error(f"Erreur lors de l'affichage des graphiques SHAP : {e}")

# @st.cache_data
# def predict_explanation():
#     try:
#         shap_data = requests.get(URL_API + "predict_explanation", params={"id_client": id_client})
#         st.write("get_shap_data shap_data", shap_data)
#         st.write("get_shap_data id_client", id_client)
#         if not shap_data.text:
#             st.error("Erreur lors de la récupération des données SHAP, shap_data.text est vide.")
#             return None  # Retournez None si la réponse est vide
#         # shap_data = shap_data.json()
#         return shap_data
#     except ValueError:
#         st.error("Erreur lors de la récupération des données SHAP.")
#         return None

# @st.cache_data
# def generate_shap_summary_plot(shap_data):
#     shap_values = pd.DataFrame(shap_data["data"])
#     data_client = pd.DataFrame(shap_data["columns"])
#     plt.figure()
#     shap.summary_plot(shap_values, data_client)
#     return plt


# Fonction pour charger la prédiction de risque de faillite
# @st.cache_data  # Utilisation de la fonction de cache de Streamlit pour éviter des appels répétés à l'API
# def load_prediction(id_client):
#     # Envoie une requête à l'API pour récupérer la prédiction de risque de faillite pour le client sélectionné
#     prediction = requests.get(URL_API + "predict", params={"id_client": id_client})
    
#     # Transforme la réponse en format JSON en utilisant .json()
#     prediction = prediction.json()
    
#     # Renvoie la probabilité de risque de faillite du client (seconde valeur du tableau JSON, index 1)
#     return prediction[1]
@st.cache_data
def load_prediction(id_client):
    try:
        prediction = requests.get(URL_API + "predict", params={"id_client": id_client})
        prediction = prediction.json()
        if prediction:
            return prediction[1]
    except ValueError:
        st.error("Erreur lors de la récupération de la prédiction.")
    return 0

# @st.cache_data
# def load_prediction(id_client):
#     prediction = requests.get(URL_API + "predict", params={"id_client": id_client})

#     # Vérifier si la réponse est nulle ou vide
#     if prediction is None:# or not prediction.text:
#         # Réponse vide ou nulle, renvoyer 0
#         return 0
    
#     try:
#         # Tenter de décoder la réponse en JSON
#         prediction_data = prediction.json()
        
#         # Vérifier si la réponse est un dictionnaire contenant la prédiction
#         if isinstance(prediction_data, dict) and 'prediction' in prediction_data:
#             return prediction_data['prediction']
#         else:
#             # La réponse ne contient pas de clé "prediction"
#             return 0
#     except json.JSONDecodeError:
#         # Erreur de décodage JSON
#         return 0

# Fonction pour charger les dossiers similaires
@st.cache_data  # Utilisation de la fonction de cache de Streamlit pour éviter des appels répétés à l'API
def load_voisins():
    try:
        # Envoie une requête à l'API pour récupérer les dossiers similaires au client sélectionné
        voisins = requests.get(URL_API + "load_voisins", params={"id_client": id_client})
        
        # Transforme la réponse en dictionnaire Python
        voisins = json.loads(voisins.content.decode("utf-8"))
        
        # Transforme le dictionnaire en un DataFrame pour une meilleure manipulation
        voisins = pd.DataFrame.from_dict(voisins).T
        
        # Extraction de la colonne "TARGET" (solvabilité) pour plus de lisibilité
        target = voisins["TARGET"]
        # Suppression de la colonne "TARGET" du DataFrame (afin d'éviter les redondances)
        voisins.drop(labels=["TARGET"], axis=1, inplace=True)
        # Réinsère la colonne "TARGET" en première position dans le DataFrame
        voisins.insert(0, "TARGET", target)
        
        # Renvoie le DataFrame contenant les dossiers similaires
        return voisins
    except ValueError:
        st.error("Erreur lors de la récupération des voisins.")

# def load_voisins():
#     # Envoie une requête à l'API pour récupérer les dossiers similaires au client sélectionné
#     voisins = requests.get(URL_API + "load_voisins", params={"id_client": id_client})
    
#     # Vérifiez si la réponse est vide ou nulle
#     if not voisins.text:
#         return {}  # Retournez un dictionnaire vide ou une autre valeur par défaut
    
#     # Transforme la réponse en dictionnaire Python
#     voisins = json.loads(voisins.content.decode("utf-8"))
    
#     # Transforme le dictionnaire en un DataFrame pour une meilleure manipulation
#     voisins = pd.DataFrame.from_dict(voisins).T
    
#     # Extraction de la colonne "TARGET" (solvabilité) pour plus de lisibilité
#     target = voisins["TARGET"]
#     # Suppression de la colonne "TARGET" du DataFrame (afin d'éviter les redondances)
#     voisins.drop(labels=["TARGET"], axis=1, inplace=True)
#     # Réinsère la colonne "TARGET" en première position dans le DataFrame
#     voisins.insert(0, "TARGET", target)
    
#     # Renvoie le DataFrame contenant les dossiers similaires
#     return voisins



if __name__ == "__main__":
    main()
