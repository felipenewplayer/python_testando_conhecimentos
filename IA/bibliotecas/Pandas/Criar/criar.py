import pandas as pd
import matplotlib.pyplot as plt

dados = pd.read_csv("C:/Users/felip/OneDrive/Área de Trabalho/Felipe/TI/Programação/PHYTON/Python2025/OpensXl/athlete_events.csv")

dados.rename(columns={'Name':'Nome', 'Sex':'Sexo','Age':'Idade'}, inplace=True)

dados.drop('ID', axis=1, inplace=True,)
dados.drop('Event', axis=1, inplace=True)

dados.hist(column='Idade', bins=10)
print(dados[0:5])
plt.show()
