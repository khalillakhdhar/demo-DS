# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:39:00 2024

@author: khali
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data=pd.read_csv('./weather_data.csv')
X = data[['Humidity']]
y = data['Temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Coefficient de la pente (b1) : {model.coef_[0]}")
print(f"Ordonnée à l'origine (b0) : {model.intercept_}")
comparison = pd.DataFrame({'Real Value': y_test, 'Predicted Value': y_pred})
print("\nComparaison des valeurs réelles et prédites :")
print(comparison)
plt.scatter(X_test, y_test, color='blue', label='Données réelles')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Prédictions')
plt.xlabel('Humidité (%)')
plt.ylabel('Température (°C)')
plt.legend()
plt.show()