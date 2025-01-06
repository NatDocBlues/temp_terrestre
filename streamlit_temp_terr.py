import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import plot_importance
from sklearn.linear_model import Lasso, LinearRegression
import plotly.express as px
import joblib
# import models

st.set_page_config(
    page_title="Température Terrestre",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ DATAFRAME & DONNEES -------------------------------------------
Temp_CO2_Glob = pd.read_csv("https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv")

temp = Temp_CO2_Glob[['Year', 'Glob', 'NHem', 'SHem']]
temp_reduit = temp.loc[(temp['Year'] >= 1900) & (temp['Year'] <= 2022)]

CO2_evol = pd.read_csv("https://nyc3.digitaloceanspaces.com/owid-public/data/co2/owid-co2-data.csv")
CO2 = CO2_evol[['year', 'country', 'population', 'co2', 'co2_per_capita', 'cumulative_co2', 'total_ghg',
                'co2_per_gdp', 'coal_co2', 'flaring_co2', 'gas_co2', 'oil_co2', 'cement_co2',
                'land_use_change_co2']]

CO2_continents = CO2.loc[(CO2['country'] == 'Africa') |
                         (CO2['country'] == 'Europe') |
                         (CO2['country'] == 'North America') |
                         (CO2['country'] == 'South America') |
                         (CO2['country'] == 'Asia') |
                         (CO2['country'] == 'Oceania')]

CO2_continents_reduit = CO2_continents.loc[CO2_continents['year'] >= 1900]
CO2_continents_reduit2 = CO2_continents_reduit.drop('total_ghg', axis = 1)
CO2_Glob = CO2_continents_reduit2.groupby('year').sum()

Temp_CO2_Glob = pd.merge(CO2_Glob, temp_reduit, left_on = 'year', right_on = 'Year')
Temp_CO2_Glob = Temp_CO2_Glob.set_index('Year')
Temp_CO2_Glob = Temp_CO2_Glob.drop('country', axis = 1)
Temp_CO2_Glob = Temp_CO2_Glob.rename(columns={
    'Glob': 'Anomalie Température Globale (°C)',
    'NHem': 'Anomalie Température Hémisphère Nord (°C)',
    'SHem': 'Anomalie Température Hémisphère Sud (°C)',
    'population': 'Population totale',
    'co2': 'Émissions de CO2 (Mt)',
    'co2_per_capita': 'Émissions de CO2 par Habitant (t/hab)',
    'cumulative_co2': 'CO2 Cumulé (Mt)',
    'co2_per_gdp' : 'CO2 par PIB',
    'coal_co2' : 'CO2 lié au charbon',
    'flaring_co2' : 'CO2 lié au torchage',
    'gas_co2' : 'CO2 lié au gaz',
    'oil_co2' : 'CO2 lié au pétrol',
    'cement_co2' : 'CO2 lié au ciment',
    'land_use_change_co2' : "CO2 lié au changement d'affectation des terres"})
# ---------------------------------------------------------

# --------------- JEU D'ENTRAINEEMNT ----------------------
feats = Temp_CO2_Glob.drop(['Anomalie Température Globale (°C)', 'Anomalie Température Hémisphère Nord (°C)',
                            'Anomalie Température Hémisphère Sud (°C)'], axis=1)
target = Temp_CO2_Glob['Anomalie Température Globale (°C)']
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state = 42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- INTERFACE STREAMLIT ------------
st.title("🌍 Température terrestre : Analyse et Modélisation")
st.sidebar.title("Sommaire")
pages=["Introduction", "Exploration des données", "Data Visualization", "Modélisation", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.title("Contacts")
st.sidebar.write('Nathanael AOUIZERAT')
st.sidebar.write('Nicolas BINAUX')
st.sidebar.write('Guillaume LEBRUN')
st.sidebar.write('Alexandre MAGGI')

# -------------- PAGE 0 ------------------------------------
if page == pages[0]:
    st.write("## Contexte")
    st.write(
    """
    Le réchauffement climatique est un phénomène documenté depuis plus de 50 ans, avec des relevés de températures et des observations scientifiques à l’échelle mondiale.
    Ses manifestations sont désormais visibles et incontestables : augmentation des vagues de chaleur, fonte accélérée des glaciers, montée du niveau des mers et bouleversements des écosystèmes. 
    Ces changements, largement attribués aux émissions de gaz à effet de serre issues des activités humaines, soulèvent des enjeux cruciaux pour l’avenir de la planète.
    """
)  
    st.write("### 100% des zones de la planète sont touchées")
    st.write(
        """
        Dans ce contexte, comprendre et anticiper l’évolution des températures est essentiel et nécessite une compréhension approfondie des tendances futures. Les impacts du réchauffement sur les sociétés, les écosystèmes et les économies rendent indispensables des outils capables de modéliser et d’anticiper les évolutions à venir.
        
        En s’appuyant sur des données climatiques collectées à l’échelle mondiale, ce travail vise à concevoir un modèle prédictif pour estimer les tendances futures.
        """
    )        
    st.write("## Objectifs")
    st.markdown(
    """
    1. **Comprendre les datasets de données climatiques :**  
       - Explorer et comprendre les deux datasets.  
       - Explorer les variables disponibles et évaluer les éventuelles limites des données (valeurs manquantes, complétudes, biais historiques).
    """
)
    st.write("")  # Ajout d'une ligne vide
    st.markdown(
    """
    2. **Analyse exploratoire :**  
       - Étudier les anomalies de température globale et régionale (évolution des hémisphères Nord/Sud).  
       - Analyser les tendances des émissions de CO2 (globale, par habitant, par pays).  
       - Détecter les corrélations entre émissions de CO2 et anomalies climatiques.
    """
)
    st.write("")  # Ajout d'une ligne vide
    st.markdown(
    """
    3. **Nettoyage et pré-processing :**  
       - Traitement des valeurs manquantes, non pertinentes ou redondantes.  
       - Fusion éventuelle des datasets.  
       - Agrégation des données lorsque c’est possible pour simplifier l’analyse globale.  
       - Création de nouvelles variables si besoin.
    """
)
    st.write("")  # Ajout d'une ligne vide
    st.markdown(
    """
    4. **Modélisation :**  
       - Test sur plusieurs algorithmes (Régression linéaire, Lasso Regression, Random Forest, XGBoost).  
       - Évaluation des performances à l’aide de métriques (R², RMSE, MAE) et sélection du modèle.
    """
)



# --------------- PAGE 1 -----------------------------------
if page == pages[1] : 
  st.write("## Exploration des données")
  st.write('Extrait du dataset')
  st.dataframe(Temp_CO2_Glob.head(10))
  st.write('Dimensions du dataset')
  st.write(Temp_CO2_Glob.shape)
  st.write('Description du dataset')
  st.dataframe(Temp_CO2_Glob.describe())
  if st.checkbox("Afficher les NA") :
    st.dataframe(Temp_CO2_Glob.isna().sum())

# --------------- PAGE 2 -----------------------------------
if page == pages[2]:
    st.write("## Data Visualization")
  
    # Chargement des données
    try:
        temperature_data = pd.read_csv("https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv")
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        st.stop()

    # ---- Section : Évolution des anomalies globales ----
    st.write("### 1. Évolution des anomalies de températures globales")
    st.write("""
    Ce graphique montre l'évolution des anomalies de température globale de 1880 à 2023.
    ##### Observations principales :
  - Une relative stabilité jusqu’aux années 1940.
  - Une augmentation notable à partir des années 1980, avec une accélération dans les années 2000.
  - Les températures actuelles dépassent souvent une anomalie de +1°C, ce qui est significatif.
  - Cette tendance reflète l'impact des activités humaines, comme l'industrialisation et l'utilisation de combustibles fossiles.
    """)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(
        temperature_data["Year"],
        temperature_data["Glob"],
        label="Anomalies de température Globale",
        color="blue"
    )
    plt.xlabel("Année")
    plt.ylabel("Température (°C)")
    plt.title("Évolution des anomalies de températures globales (1880-2023)")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)
    st.write("--------------------")
    # ---- Section : Comparaison des anomalies Nord/Sud ----
    st.write("### 2. Évolution des anomalies de températures - Hémisphère Nord et Sud")
    st.write("""
    Ce graphique compare les anomalies de températures entre l'hémisphère Nord et l'hémisphère Sud.
    ##### Observations principales :
  - L’hémisphère Nord (en rouge) montre une augmentation plus rapide des températures par rapport à l’hémisphère Sud (en vert).
  - L’écart s’élargit surtout après les années 1950, probablement à cause de :
    - Une concentration plus importante de terres émergées dans l’hémisphère Nord.
    - Une intensité plus élevée des activités humaines dans cette région.
  - Ces différences régionales illustrent des impacts variés du changement climatique selon les zones géographiques.
    """)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(
        temperature_data["Year"],
        temperature_data["NHem"],
        label="Hémisphère Nord",
        color="red"
    )
    plt.plot(
        temperature_data["Year"],
        temperature_data["SHem"],
        label="Hémisphère Sud",
        color="green"
    )
    plt.xlabel("Année")
    plt.ylabel("Température (°C)")
    plt.title("Comparaison des anomalies de températures entre l'hémisphère Nord et Sud (1880-2023)")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)
    st.write("--------------------")
    # ---- Section : Distribution des anomalies ----
    st.write("### 3. Distribution des anomalies de température globale")
    st.write("""
    Ce boxplot montre la distribution des anomalies de températures globales sur toute la période d'analyse.
    ##### Observations principales :
    - La médiane (ligne au centre de la boîte) est proche de 0, reflétant une relative stabilité pendant une grande partie de la période.
    - Une asymétrie apparaît vers des valeurs positives, montrant un réchauffement global dans les années récentes.
    - Les points aberrants (outliers) représentent des années exceptionnelles où les anomalies étaient particulièrement élevées.
    """)
    fig = plt.figure(figsize=(6, 4))
    sns.boxplot(x=temperature_data['Glob'], color="blue")
    plt.title("Distribution des anomalies de températures globales")
    plt.xlabel("Température (°C)")
    st.pyplot(fig)
    st.write("--------------------")
    # ---- Section : Comparaison des anomalies Nord/Sud ----
    st.write("### 4. Évolution des émissions de CO2 par habitant pour des pays sélectionnés")
    st.write("""
    Ce graphique illustre les tendances des émissions de CO2 par habitant (en tonnes) pour plusieurs pays au fil des années. Voici les principaux points d'analyse :
    ##### Observations principales :
    1. **États-Unis (en bleu)** :
        - Les États-Unis ont historiquement les émissions par habitant les plus élevées, avec un pic autour de 20 tonnes par habitant dans les années 1970.
        - Depuis, une tendance à la baisse est observée, grâce à des politiques de transition énergétique et à une meilleure efficacité énergétique.
    2. **Chine (en orange)** :
        - La Chine montre une croissance marquée des émissions par habitant à partir des années 1990, reflétant une industrialisation rapide.
        - Les émissions par habitant restent inférieures à celles des États-Unis, mais la tendance est croissante.
    3. **Inde (en vert)** :
        - L'Inde affiche des émissions par habitant très faibles, même avec sa forte croissance économique.
        - Cela souligne les disparités dans les contributions au CO2 entre les pays industrialisés et en développement.
    4. **Russie (en rouge)** :
        - Les émissions par habitant de la Russie connaissent une forte baisse après la chute de l'Union soviétique dans les années 1990, mais elles se stabilisent ensuite.
    5. **Brésil (en violet)** :
        - Le Brésil a des émissions modérées et relativement stables par habitant, probablement en raison d'une utilisation plus importante des énergies renouvelables comme l'hydroélectricité.
    6. **France (en marron)** :
        - La France montre une baisse constante des émissions par habitant grâce à sa transition vers le nucléaire et les énergies renouvelables.
    """)
    # Filtrage des données pour les pays spécifiques
    countries = ["United States", "China", "India", "Russia", "Brazil", "France"]
    country_data = CO2_evol[CO2_evol["country"].isin(countries)]
    # Gestion des données manquantes
    country_data = country_data.dropna(subset=["year", "co2_per_capita"])
    # Initialisation de la figure avec plt.subplots()
    fig, ax = plt.subplots(figsize=(14, 7))
    # Tracé des courbes pour chaque pays
    for country in countries:
        country_subset = country_data[country_data["country"] == country]
        ax.plot(
            country_subset["year"],
            country_subset["co2_per_capita"],
            label=country
        )
    # Ajout des labels, titre, et légende
    ax.set_xlabel("Année")
    ax.set_ylabel("Émissions de CO2 par habitant (tonnes)")
    ax.set_title("Évolution des émissions de CO2 par habitant pour des pays sélectionnés")
    ax.legend()
    ax.grid(True)
    # Affichage du graphique dans Streamlit
    st.pyplot(fig)
    st.write("""
    - Ce graphique met en évidence des disparités majeures entre les pays industrialisés et les pays émergents.
    - Les pays développés comme les États-Unis et la France ont amorcé une réduction de leurs émissions, tandis que les pays émergents comme la Chine et l'Inde augmentent leurs émissions en raison de la croissance industrielle.
    - Ces tendances illustrent les défis mondiaux pour atteindre les objectifs climatiques, notamment une transition équitable vers des économies à faible émission de carbone.
    """)
    st.write("--------------------")
    # ---- Matrice de corrélation ----
    st.write("### 5. Matrice de corrélation")
    st.write("""
    Cette matrice de corrélation montre les relations entre les différentes variables climatiques et socio-économiques.
    Les corrélations mesurent la force et la direction de la relation entre deux variables :
        - **1.0** : Corrélation positive parfaite.
        - **-1.0** : Corrélation négative parfaite.
        - **0.0** : Aucune corrélation.
    ##### Observations principales :
    1. Les **émissions de CO2 totales** montrent une forte corrélation positive avec :
        - Les anomalies de température globale (**0.94**).
        - Les anomalies de température des hémisphères Nord (**0.93**) et Sud (**0.95**).
     - Cela indique que l'augmentation des émissions de CO2 est directement liée au réchauffement global.
    2. La **population totale** est fortement corrélée avec les émissions de CO2 (**1.0**) :
        - Plus la population est importante, plus les émissions de CO2 augmentent en raison des besoins en énergie et des activités industrielles.
    3. Les émissions liées à différentes sources d'énergie (gaz, pétrole, charbon) sont fortement corrélées entre elles :
        - Par exemple, **CO2 lié au charbon** et **CO2 lié au pétrole** ont une corrélation de **0.99**, reflétant l'utilisation simultanée de ces sources d'énergie fossile.
    4. Les **anomalies de température globale** présentent une forte corrélation avec :
        - Les émissions de CO2 totales (**0.94**).
        - Les anomalies de température dans les hémisphères Nord (**0.98**) et Sud (**0.95**), ce qui montre un effet global mais plus prononcé dans l’hémisphère Nord en raison de l’intensité des activités humaines.
    5. Le **CO2 lié au changement d’affectation des terres** a une faible corrélation avec les anomalies de température (**-0.37 à -0.41**) :
        - Cela peut s'expliquer par la complexité des interactions entre la déforestation et les émissions directes ou indirectes.
    Ces observations soulignent l'importance de la réduction des émissions de CO2 pour limiter les anomalies climatiques à l'échelle mondiale.
    """)
    # Calcul de la matrice de corrélation pour les colonnes numériques uniquement
    numeric_data = Temp_CO2_Glob.select_dtypes(include=['float64', 'int64'])
    numeric_data = numeric_data.fillna(0)  # Remplacement des valeurs manquantes par 0
    correlation_matrix = numeric_data.corr()    # Visualisation de la matrice de corrélation avec Seaborn
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,                # Affiche les valeurs dans les cellules
        fmt=".2f",                 # Format des valeurs avec deux décimales
        cmap="coolwarm",           # Palette de couleurs
        linewidths=0.5,            # Ajoute des séparateurs entre les cellules
        cbar_kws={"shrink": 0.8}   # Ajuste la taille de la barre de couleurs
    )
    plt.title("Matrice de corrélation des variables climatiques et socio-économiques", fontsize=16, pad=20)
    plt.xticks(fontsize=12, rotation=45, ha='right')  # Rotation etalignement des étiquettes sur l'axe x
    plt.yticks(fontsize=12)                           # Taille des étiquettes sur l'axe y
    st.pyplot(fig)
    st.write("--------------------")

# --------------- PAGE 3 -----------------------------------
if page == pages[3] :
    st.write("## Modélisation")
    choix = ['XGBOOST', 'Lasso', 'Regression Linéaire', 'Random Forest']
    option = st.selectbox('### Choix du modèle', choix)
    st.write('### Le modèle choisi est :', option)

    model = joblib.load(f"{option}.joblib")
    # model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write("R2 jeu d'entrainement:", model.score(X_train_scaled, y_train))
    st.write("R2 jeu de test:", model.score(X_test_scaled, y_test))    
    st.write('RMSE:', rmse)
    # st.write('R2', r2)
    st.write('MSE', mse)
    st.write('MAE', mae)
    
    def plot_feature_importance(feature_importance, title):
        fig_fi = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', title=title)
        st.plotly_chart(fig_fi)

    match option :
        case "Lasso":
            lasso_importances = pd.DataFrame({
                'Feature': feats.columns,
                'Importance': model.coef_
            }).sort_values(by='Importance', ascending=False)
            plot_feature_importance(lasso_importances, "Importance des caractéristiques - Lasso Regression")       
        
        case "XGBOOST":
            xgb_importances = pd.DataFrame({
                'Feature': feats.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            plot_feature_importance(xgb_importances, "Importance des caractéristiques - XGBoost")        

        case "Regression Linéaire":
            linear_importances = pd.DataFrame({
                'Feature': feats.columns,
                'Importance': model.coef_
            }).sort_values(by='Importance', ascending=False)
            plot_feature_importance(linear_importances, "Importance des caractéristiques - Régression Linéaire")
            
        case "Random Forest":
            rf_importances = pd.DataFrame({
                'Feature': feats.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            plot_feature_importance(rf_importances, "Importance des caractéristiques - Random Forest")
        
    st.write('### Simulation')

    col1a, col2a = st.columns(2)
    with col1a:
        slider_population  = st.slider('Population',0,100,50)
        slider_co2  = st.slider('CO2',0,100,50)
        slider_co2_per_capita = st.slider('CO2 per capita',0,100,50)
        slider_cumulative_co2 = st.slider('Cumulative CO2',0,100,50)
        slider_co2_per_gdp = st.slider('CO2 per GDP',0,100,50)
        slider_coal_co2 = st.slider('Coal CO2',0,100,50)
    
    with col2a:
        slider_flaring_co2 = st.slider('Flaring CO2',0,100,50)
        slider_gas_co2 = st.slider('Gas CO2',0,100,50)
        slider_oil_co2 = st.slider('Oil CO2',0,100,50)
        slider_cement_co2 = st.slider('Cement CO2',0,100,50)
        slider_land_use_change_co2  = st.slider('Land use change CO2',0,100,50)
    

# ------------------ PAGE : CONCLUSION -------------------
if page == pages[4]:
    st.write("## Conclusion")

    st.write("### Difficultés rencontrées")
    st.write(
        """
        - **Valeurs manquantes**  
          Les lacunes dans les données historiques (notamment avant 1900) ont nécessité des imputation et un ajustement rigoureux pour minimiser l’impact sur les analyses.
          Les données avant 1900 ont été exclues grâce à un filtrage explicite pour éviter d’incorporer des données incomplètes.
              
        - **Disparités régionales**  
          Les variations régionales importantes (entre continents et pays) ont complexifié l’interprétation des tendances globales.
          Les émissions de CO2 ont été regroupées par continent pour une analyse simplifiée.
          Les anomalies de température ont été comparées entre l’hémisphère Nord et Sud.

        
        - **Modélisation complexe**  
          L’ajustement des hyperparamètres pour certains modèles (notamment XGBoost et Random Forest) a demandé des itérations longues.  
          Des risques d’overfitting ont été identifiés pour certains modèles avancés, bien que des validations croisées aient permis de les atténuer.
        
        """
    )
    st.write("### Réponses aux objectifs")

    st.write(
        """
       1. **Comprendre les datasets de données climatiques :**  
       - Les datasets de la NASA (anomalies de température) et de Global Change Data Lab (émissions de CO2) ont été analysés, intégrés et fusionnés efficacement.  
       - Les variables clés ont été identifiées et enrichies, et les limites des données (valeurs manquantes, biais historiques) ont été prises en compte.
       """
    )
    st.write("")  # Ajout d'une ligne vide

    st.write(
        """
    2. **Analyse exploratoire :**  
       - Les tendances des anomalies de température globale et régionale ont été mises en évidence, montrant une accélération depuis les années 1950.
       - Les disparités régionales (hémisphères Nord/Sud, continents) et les contributions des principaux pays émetteurs de CO2 (Chine, USA, Inde) ont été analysées.
       - Les corrélations fortes (≈ 0,95) entre émissions de CO2 et anomalies climatiques ont été confirmées.
       """
    )
    st.write("")  

    st.write(
        """
    3. **Nettoyage et pré-processing :**  
       - Les données ont été nettoyées avec succès, fusionnées sur une clé commune (année) et enrichies avec de nouvelles variables (variation annuelle des émissions, décalage temporel).
       - L’agrégation par continent a simplifié l’analyse tout en préservant les grandes tendances régionales.
       """
    )
    st.write("")  

    st.write(
        """
    4. **Modélisation :**  
       - Plusieurs algorithmes ont été testés (Régression linéaire, Lasso, Random Forest, XGBoost).
       - Le modèle Lasso Regression a été sélectionné comme le plus performant grâce à un R² proche de 1 et une interprétabilité claire, confirmant son aptitude à prédire les anomalies de température.
       """
    )
    st.write("")  

    st.write("### Résultats compilés :")
    results_data = []
    for model_name in ["Regression Linéaire", "Lasso", "XGBOOST", "Random Forest"]:
        try:
            # Charger chaque modèle sauvegardé
            model = joblib.load(f"{model_name}.joblib")
            y_pred = model.predict(X_test_scaled)
 
           # Collecter les métriques pour chaque modèle
            results_data.append({
                "Modèle": model_name,
                "R² (Train)": model.score(X_train_scaled, y_train),
                "R² (Test)": model.score(X_test_scaled, y_test),
                "MSE (Test)": mean_squared_error(y_test, y_pred),
                "RMSE (Test)": root_mean_squared_error(y_test, y_pred),
                "MAE (Test)": mean_absolute_error(y_test, y_pred)
            })
        except Exception as e:
          st.error(f"Erreur avec le modèle {model_name} : {e}")
 
    # Créer un DataFrame pour afficher les résultats
    results_df = pd.DataFrame(results_data)
 
    # Créer un DataFrame fondu pour permettre l'animation
    df_melted = results_df.melt(id_vars="Modèle", var_name="Métrique", value_name="Valeur")
 
    # Définir une plage dynamique maximale pour chaque métrique
    range_dict = {
        "R² (Train)": [0, 1.5],
        "R² (Test)": [0, 1.5],
        "MSE (Test)": [0, df_melted[df_melted["Métrique"] == "MSE (Test)"]["Valeur"].max() * 1.2],
        "RMSE": [0, df_melted[df_melted["Métrique"] == "RMSE"]["Valeur"].max() * 1.2],
        "MAE": [0, df_melted[df_melted["Métrique"] == "MAE"]["Valeur"].max() * 1.2],
    }
 
    # Créer le graphique animé avec des plages spécifiques pour chaque métrique
    fig = px.bar(
        df_melted,
        x="Modèle",
        y="Valeur",
        color="Modèle",
        animation_frame="Métrique",
        title="Évolution des performances par métrique",
        labels={"Valeur": "Valeur", "Modèle": "Modèle", "Métrique": "Métrique"},
    )
 
    # Ajouter des plages dynamiques pour chaque métrique
    fig.update_layout(
        yaxis=dict(range=[0, max(range_dict.values(), key=lambda x: x[1])[1]])
    )
 
    # Afficher les valeurs sur les barres
    fig.update_traces(texttemplate="%{y:.3f}", textposition="outside")
 
    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)
 
    # Ajouter un texte explicatif
    st.write("### Analyse des performances des modèles")
  
    # Transformation en format long
    results_melted = results_df.melt(id_vars="Modèle", var_name="Métrique", value_name="Valeur")        
 
    # Afficher les résultats sous forme de tableau
    st.dataframe(results_df, use_container_width=True)

    # Créer les heatmaps à partir des résultats
    st.write("### Heatmap des résultats")
 
    # Calcul des rangs (inversion pour les métriques où une valeur plus basse est meilleure)
    results_ranked = results_df.copy()
    for metric in ["MSE (Test)", "RMSE (Test)", "MAE (Test)"]:
        results_ranked[metric] = results_df[metric].rank(ascending=True)  # Plus bas est mieux
    for metric in ["R² (Train)", "R² (Test)"]:
        results_ranked[metric] = results_df[metric].rank(ascending=False)  # Plus haut est mieux
 
    # Transposer pour les heatmaps
    values_heatmap = results_df.set_index("Modèle").T
    ranked_heatmap = results_ranked.set_index("Modèle").T
 
    # Afficher les deux heatmaps côte à côte
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Valeurs réelles")
        fig_values = px.imshow(
            values_heatmap,
            labels={"x": "Modèle", "y": "Métrique", "color": "Valeur"},
            title="Heatmap des valeurs réelles",
            text_auto=True,
            color_continuous_scale="Blues",
            width=500,
            height=500
        )
        st.plotly_chart(fig_values)
 
    with col2:
        st.write("#### Classements")
        fig_ranks = px.imshow(
            ranked_heatmap,
            labels={"x": "Modèle", "y": "Métrique", "color": "Classement"},
            title="Heatmap des classements",
            text_auto=True,
            color_continuous_scale="Greens",
            width=500,
            height=500
        )
        st.plotly_chart(fig_ranks)

    

    st.write("""
    Ce graphique permet de visualiser les performances des différents modèles selon plusieurs indicateurs :
    - **R² (Train/Test)** : Mesure la capacité des modèles à expliquer la variance des données sur les ensembles d'entraînement et de test.
    - **MSE (Test)** : Moyenne des erreurs quadratiques des prédictions, plus elle est basse, mieux c'est.
    - **RMSE** : Racine de l'erreur quadratique moyenne, utile pour interpréter les erreurs dans l'unité d'origine.
    - **MAE** : Moyenne des erreurs absolues, représentant l'écart moyen des prédictions par rapport aux valeurs réelles.

    Les modèles ayant une aire plus étendue indiquent une meilleure performance globale. Par exemple :
    - **XGBoost** se distingue par une précision élevée sur les jeux de test.
    - **Régression Linéaire** montre une bonne généralisation mais des limites pour les erreurs (MAE et RMSE).
    """)

    st.write("## Conclusion générale")
    st.write("### Modèle sélectionné : XGBoost")
    st.write(
    """
    Après analyse, le modèle **XGBoost** a été sélectionné comme le meilleur modèle pour prédire les anomalies de température. Voici pourquoi :
    """
    )

    # **1. Performances supérieures**
    st.write("#### 1. Performances supérieures")
    st.write(
    """
    - XGBoost obtient les meilleurs scores parmi les modèles testés :  
      - **R² sur les données de test : 0.896**  
        Explique près de **90 %** de la variance des anomalies de température.  
      - **RMSE : 0.112**  
        Montre une faible erreur quadratique moyenne des prédictions.  
      - **MAE : 0.095**  
        Indique une erreur absolue moyenne très faible.  

    - Comparé aux autres modèles :
      - Régression Linéaire : R² = 0.871, RMSE = 0.125, MAE = 0.105.  
      - Lasso Regression : R² = 0.829, RMSE = 0.144, MAE = 0.122.  
      - Random Forest : R² = 0.887, RMSE = 0.117, MAE = 0.099.  
      
    Ces résultats confirment que XGBoost est le modèle le plus précis et fiable.
    """
)

    # **2. Adaptabilité aux données complexes**
    st.write("#### 2. Adaptabilité aux données complexes")
    st.write(
    """
    - XGBoost capture mieux les relations complexes et non linéaires entre les variables (émissions de CO2, population, etc.) :  
      - Avec un **R² de 0.896** sur les données de test, il surpasse largement les modèles linéaires.  
      - Les interactions entre les émissions par habitant, la population et d'autres variables sont bien modélisées.  
      
    Cela lui permet de mieux expliquer les tendances que des modèles simples comme la Régression Linéaire (R² = 0.871).
    """
    )

    # **3. Robustesse face à l'overfitting**
    st.write("#### 3. Robustesse face à l'overfitting")
    st.write(
    """
    - XGBoost utilise une régularisation intégrée qui réduit le risque d'overfitting :  
      - **R² sur les données d'entraînement : 0.999**  
        Montre un ajustement quasi parfait.  
      - **R² sur les données de test : 0.896**  
        La faible différence entre ces scores montre une excellente capacité de généralisation.  

    - En comparaison :
      - Random Forest, bien qu'efficace, montre un R² légèrement inférieur sur les données de test (0.887).  
      - Les modèles linéaires, bien qu'ils généralisent mieux, ont des performances globales inférieures.  

    XGBoost offre ainsi un bon équilibre entre précision et robustesse.
    """
    )

    # ------------------- CHARGEMENT DU MODÈLE XGBOOST ET SCÉNARIOS -------------------
# Charger le modèle XGBoost
    model = joblib.load("XGBOOST.joblib")
 
    st.write("## Prédictions avec le modèle XGBoost")
 
# Scénarios
    scenarios = {
    "Scénario actuel": {
        "Population totale": X_test["Population totale"].max(),
        "Émissions de CO2 (Mt)": X_test["Émissions de CO2 (Mt)"].max(),
        "Émissions de CO2 par Habitant (t/hab)": X_test["Émissions de CO2 par Habitant (t/hab)"].max(),
        "CO2 Cumulé (Mt)": X_test["CO2 Cumulé (Mt)"].max(),
        "CO2 par PIB": X_test["CO2 par PIB"].max(),
        "CO2 lié au charbon": X_test["CO2 lié au charbon"].max(),
        "CO2 lié au torchage": X_test["CO2 lié au torchage"].max(),
        "CO2 lié au gaz": X_test["CO2 lié au gaz"].max(),
        "CO2 lié au pétrol": X_test["CO2 lié au pétrol"].max(),
        "CO2 lié au ciment": X_test["CO2 lié au ciment"].max(),
        "CO2 lié au changement d'affectation des terres": X_test["CO2 lié au changement d'affectation des terres"].max()
    },
    "Population double, CO₂ divisé par 2": {
        "Population totale": X_test["Population totale"].max() * 2,
        "Émissions de CO2 (Mt)": X_test["Émissions de CO2 (Mt)"].max() / 2,
        "Émissions de CO2 par Habitant (t/hab)": X_test["Émissions de CO2 par Habitant (t/hab)"].max() / 2,
        "CO2 Cumulé (Mt)": X_test["CO2 Cumulé (Mt)"].max() / 2,
        "CO2 par PIB": X_test["CO2 par PIB"].max() / 2,
        "CO2 lié au charbon": X_test["CO2 lié au charbon"].max() / 2,
        "CO2 lié au torchage": X_test["CO2 lié au torchage"].max() / 2,
        "CO2 lié au gaz": X_test["CO2 lié au gaz"].max() / 2,
        "CO2 lié au pétrol": X_test["CO2 lié au pétrol"].max() / 2,
        "CO2 lié au ciment": X_test["CO2 lié au ciment"].max() / 2,
        "CO2 lié au changement d'affectation des terres": X_test["CO2 lié au changement d'affectation des terres"].max() / 2
    },
    "Réduction massive des émissions": {
        "Population totale": X_test["Population totale"].max(),
        "Émissions de CO2 (Mt)": X_test["Émissions de CO2 (Mt)"].max() * 0.2,
        "Émissions de CO2 par Habitant (t/hab)": X_test["Émissions de CO2 par Habitant (t/hab)"].max() * 0.2,
        "CO2 Cumulé (Mt)": X_test["CO2 Cumulé (Mt)"].max() * 0.2,
        "CO2 par PIB": X_test["CO2 par PIB"].max() * 0.2,
        "CO2 lié au charbon": X_test["CO2 lié au charbon"].max() * 0.2,
        "CO2 lié au torchage": X_test["CO2 lié au torchage"].max() * 0.2,
        "CO2 lié au gaz": X_test["CO2 lié au gaz"].max() * 0.2,
        "CO2 lié au pétrol": X_test["CO2 lié au pétrol"].max() * 0.2,
        "CO2 lié au ciment": X_test["CO2 lié au ciment"].max() * 0.2,
        "CO2 lié au changement d'affectation des terres": X_test["CO2 lié au changement d'affectation des terres"].max() * 0.2
    },
    "Augmentation des énergies fossiles": {
        "Population totale": X_test["Population totale"].max(),
        "Émissions de CO2 (Mt)": X_test["Émissions de CO2 (Mt)"].max() * 1.5,
        "Émissions de CO2 par Habitant (t/hab)": X_test["Émissions de CO2 par Habitant (t/hab)"].max() * 1.5,
        "CO2 Cumulé (Mt)": X_test["CO2 Cumulé (Mt)"].max() * 1.5,
        "CO2 par PIB": X_test["CO2 par PIB"].max() * 1.5,        
        "CO2 lié au charbon": X_test["CO2 lié au charbon"].max() * 1.5,
        "CO2 lié au torchage": X_test["CO2 lié au torchage"].max() * 1.5,    
        "CO2 lié au gaz": X_test["CO2 lié au gaz"].max() * 1.5,            
        "CO2 lié au pétrol": X_test["CO2 lié au pétrol"].max() * 1.5,
        "CO2 lié au ciment": X_test["CO2 lié au ciment"].max() * 1.5,
        "CO2 lié au changement d'affectation des terres": X_test["CO2 lié au changement d'affectation des terres"].max() * 1.5
    },
    "Transition vers les énergies renouvelables": {
        "Population totale": X_test["Population totale"].max(),
        "Émissions de CO2 (Mt)": X_test["Émissions de CO2 (Mt)"].max() * 0.1,
        "Émissions de CO2 par Habitant (t/hab)": X_test["Émissions de CO2 par Habitant (t/hab)"].max() * 0.1,
        "CO2 Cumulé (Mt)": X_test["CO2 Cumulé (Mt)"].max() * 0.1,
        "CO2 par PIB": X_test["CO2 par PIB"].max() * 0.1,        
        "CO2 lié au charbon": X_test["CO2 lié au charbon"].max() * 0.1,
        "CO2 lié au torchage": X_test["CO2 lié au torchage"].max() * 0.1,          
        "CO2 lié au gaz": X_test["CO2 lié au gaz"].max() * 0.1,
        "CO2 lié au pétrol": X_test["CO2 lié au pétrol"].max() * 0.1,
        "CO2 lié au ciment": X_test["CO2 lié au ciment"].max() * 0.1,
        "CO2 lié au changement d'affectation des terres": X_test["CO2 lié au changement d'affectation des terres"].max() * 0.1
    },
    "Croissance démographique rapide": {
        "Population totale": X_test["Population totale"].max() * 3,
        "Émissions de CO2 (Mt)": X_test["Émissions de CO2 (Mt)"].max(),
        "Émissions de CO2 par Habitant (t/hab)": X_test["Émissions de CO2 par Habitant (t/hab)"].max(),
        "CO2 Cumulé (Mt)": X_test["CO2 Cumulé (Mt)"].max(),
        "CO2 par PIB": X_test["CO2 par PIB"].max(),        
        "CO2 lié au charbon": X_test["CO2 lié au charbon"].max(),
        "CO2 lié au torchage": X_test["CO2 lié au torchage"].max(),          
        "CO2 lié au gaz": X_test["CO2 lié au gaz"].max(),
        "CO2 lié au pétrol": X_test["CO2 lié au pétrol"].max(),
        "CO2 lié au ciment": X_test["CO2 lié au ciment"].max(),
        "CO2 lié au changement d'affectation des terres": X_test["CO2 lié au changement d'affectation des terres"].max()      
    },
    "Scénario optimiste : réduction globale des émissions": {
        "Population totale": X_test["Population totale"].max(),
        "Émissions de CO2 (Mt)": X_test["Émissions de CO2 (Mt)"].max() * 0.3,
        "Émissions de CO2 par Habitant (t/hab)": X_test["Émissions de CO2 par Habitant (t/hab)"].max() * 0.3,
        "CO2 Cumulé (Mt)": X_test["CO2 Cumulé (Mt)"].max() * 0.3,
        "CO2 par PIB": X_test["CO2 par PIB"].max() * 0.3,            
        "CO2 lié au charbon": X_test["CO2 lié au charbon"].max() * 0.3,
        "CO2 lié au torchage": X_test["CO2 lié au torchage"].max() * 0.3,          
        "CO2 lié au gaz": X_test["CO2 lié au gaz"].max() * 0.3,
        "CO2 lié au pétrol": X_test["CO2 lié au pétrol"].max() * 0.3,
        "CO2 lié au ciment": X_test["CO2 lié au ciment"].max() * 0.3,
        "CO2 lié au changement d'affectation des terres": X_test["CO2 lié au changement d'affectation des terres"].max() * 0.3  
    }
    }
 
# Afficher les scénarios dans une liste déroulante
    scenario_choice = st.selectbox("Choisissez un scénario", list(scenarios.keys()))
    scenario_data = pd.DataFrame([scenarios[scenario_choice]])
 
# Afficher les données du scénario
    st.write(f"### Données du scénario : {scenario_choice}")
    st.dataframe(scenario_data)
 
# Appliquer le scaling sur les données du scénario
    try:
        scenario_data_scaled = scaler.transform(scenario_data)
        predicted_anomaly = model.predict(scenario_data_scaled)
        st.write(f"🌍 **Anomalie prédite :** {predicted_anomaly[0]:.2f} °C")
    except Exception as e:
        st.error(f"Erreur : {e}")