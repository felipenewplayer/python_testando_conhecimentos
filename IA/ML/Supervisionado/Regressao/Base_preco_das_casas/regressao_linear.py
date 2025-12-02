import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
base_houses_prices = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\ML\Supervisionado\Regressao\Base_preco_das_casas\house_prices.csv')
base_houses_prices = base_houses_prices.drop('date', axis=1)

print(base_houses_prices.head())
print(base_houses_prices.describe())
print(base_houses_prices.isnull().sum())
print(base_houses_prices.corr())

figura = plt.figure(figsize=(20,20))
sns.heatmap(base_houses_prices.corr(), annot=True, cmap='viridis')
plt.show()
