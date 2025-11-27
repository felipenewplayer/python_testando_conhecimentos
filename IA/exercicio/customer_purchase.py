from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


df = pd.DataFrame({
    'Dono':['Marcos','Rafa','Felipe'],
    'Carros':['Porche','Ferrari','BMW-X10'],
    'Ano':[2021,1997,2010]
})


ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(sparse_output=False),
        ['Dono','Carros'])],
    remainder='passthrough'
)

x_encoder = ct.fit_transform(df)
print(ct.get)


