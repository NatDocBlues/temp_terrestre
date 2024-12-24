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
    'land_use_change_co2' : 'CO2 lié au changement d"affectation des terres'})
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
if page == pages[2] : 
    st.write("## Data Vizualisation")
    st.write("### Nettoyage & Pre-Processing")
    # -----
    st.write("### Evolution des anomalies de temperatures")
    fig = plt.figure()
    temperature_data = pd.read_csv("https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv")
    plt.plot(temperature_data["Year"], temperature_data["Glob"], label="Anomalies de température Globale", color="blue")
    plt.xlabel("Année")
    plt.ylabel("Température (°C)")
    plt.title("Évolution des anomalies de températures globales (1880-2023)")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

    fig = plt.figure()
    st.write("### Evolution des anomalies de temperatures - Hémisphère Nord et Sud")
    plt.plot(temperature_data["Year"], temperature_data["NHem"], label="Hémisphère Nord", color="red")
    plt.plot(temperature_data["Year"], temperature_data["SHem"], label="Hémisphère Sud", color="green")
    plt.xlabel("Année")
    plt.ylabel("Température (°C)")
    plt.title("Comparaison des anomalies de températures entre l'hémisphère nord et sud (1880-2023)")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

    fig = plt.figure()
    st.write("### DIstribution des anomalies de temperature global")
    sns.boxplot(x = temperature_data['Glob'])
    st.pyplot(fig)



# --------------- PAGE 3 -----------------------------------
if page == pages[3] :
    st.write("## Modélisation")
    choix = ['XGBOOST', 'Lasso', 'Regression Linéaire', 'Random Forest']
    option = st.selectbox('### Choix du modèle', choix)
    st.write('### Le modèle choisi est :', option)

    model = joblib.load(f"{option}.joblib")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write("R2 jeu d'entrainement:", model.score(X_train_scaled, y_train))
    st.write("R2 jeu de test:", model.score(X_test_scaled, y_test))    
    st.write('RMSE:', rmse)
    st.write('R2', r2)
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
    
    slider_population  = st.slider('Population',0,100,50)
    slider_co2  = st.slider('CO2',0,100,50)
    slider_co2_per_capita = st.slider('CO2 per capita',0,100,50)
    slider_cumulative_co2 = st.slider('Cumulative CO2',0,100,50)
    slider_co2_per_gdp = st.slider('CO2 per GDP',0,100,50)
    slider_coal_co2 = st.slider('Coal CO2',0,100,50)
    slider_flaring_co2 = st.slider('Flaring CO2',0,100,50)
    slider_gas_co2 = st.slider('Gas CO2',0,100,50)
    slider_oil_co2 = st.slider('Oil CO2',0,100,50)
    slider_cement_co2 = st.slider('Cement CO2',0,100,50)
    slider_land_use_change_co2  = st.slider('Land use change CO2',0,100,50)


    

