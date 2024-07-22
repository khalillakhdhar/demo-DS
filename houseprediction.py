# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:54:13 2024

@author: khali
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = pd.read_csv("./house_data.csv")
X = data[['SquareFeet']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#Affichage des coefficients
print(f"Coefficient de la pente (b1) : {model.coef_[0]}")
print(f"Ordonnée à l'origine (b0) : {model.intercept_}")
comparison = pd.DataFrame({'Real Value': y_test, 'Predicted Value': y_pred})
print("\nComparaison des valeurs réelles et prédites :")
print(comparison)
#Visualisation des résultats
plt.scatter(X_test, y_test, color='blue', label='Données réelles')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Prédictions')
plt.xlabel('Superficie (pieds carrés)')
plt.ylabel('Prix (milliers de dollars)')
plt.legend()
plt.show()