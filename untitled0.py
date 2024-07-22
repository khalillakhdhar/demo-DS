import pandas as pd
import numpy as np
#Génération des données
np.random.seed(0)
square_feet = np.random.randint(500, 3500, 10000)  #Superficie en pieds carrés
price = 0.05 * square_feet + np.random.normal(0, 20, 10000) #Prix en milliers de dollars
#Création du DataFrame
house_data = pd.DataFrame({
 'SquareFeet': square_feet,
 'Price': price
})
#Sauvegarde en fichier CSV
file_path = "./house_data.csv"
house_data.to_csv(file_path, index=False)
