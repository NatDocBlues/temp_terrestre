import pandas as pd
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

# ------------------ DATAFRAME & DONNEES ---------------------------------------
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
    'population': 'Population Totale',
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

# --------------- MODELES ----------------------------------
# Fonction de visualisation des importances des caractéristiques
def plot_feature_importance(feature_importance, title):
    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', title=title)
    fig.update_layout(yaxis_title="Caractéristiques", xaxis_title="Importance", title_x=0.5)
    fig.show()

# MODELE 1 : XGBoost
feats = Temp_CO2_Glob.drop(['Anomalie Température Globale (°C)', 'Anomalie Température Hémisphère Nord (°C)',
                            'Anomalie Température Hémisphère Sud (°C)'], axis=1)
target = Temp_CO2_Glob['Anomalie Température Globale (°C)']
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state = 42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb_params = {
    'eta': [0.1, 0.3, 0.5],
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.5, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0]
}

xgb_model = xgb.XGBRegressor()
grid_search_xgb = GridSearchCV(xgb_model, param_grid=xgb_params, scoring='r2', cv=5, verbose=0)
grid_search_xgb.fit(X_train_scaled, y_train)
best_xgb = grid_search_xgb.best_estimator_
joblib.dump(best_xgb, "XGBOOST.joblib")

xgb_importances = pd.DataFrame({
    'Feature': feats.columns,
    'Importance': grid_search_xgb.best_estimator_.feature_importances_
}).sort_values(by='Importance', ascending=False)

# MODELE 2 : Lasso Regression
lasso_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
lasso_model = Lasso()
grid_search_lasso = GridSearchCV(lasso_model, param_grid=lasso_params, scoring='neg_mean_squared_error', cv=5)
grid_search_lasso.fit(X_train_scaled, y_train)
best_lasso_model = grid_search_lasso.best_estimator_
joblib.dump(best_lasso_model, "Lasso.joblib")

lasso_importances = pd.DataFrame({
    'Feature': feats.columns,
    'Importance': grid_search_lasso.best_estimator_.coef_
}).sort_values(by='Importance', ascending=False)

# MODELE 3 : Régression Linéaire
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
joblib.dump(linear_model, "Regression Linéaire.joblib")

linear_importances = pd.DataFrame({
    'Feature': feats.columns,
    'Importance': linear_model.coef_
}).sort_values(by='Importance', ascending=False)

# MODELE 4 : Random Forest
rf_params = {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
rf_model = RandomForestRegressor(**rf_params, random_state=42)
rf_model.fit(X_train_scaled, y_train)
joblib.dump(rf_model, "Random Forest.joblib")

rf_importances = pd.DataFrame({
    'Feature': feats.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

