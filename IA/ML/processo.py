import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# 1️⃣ Carregar dados
df = pd.read_csv(r'C:\Users\felip\OneDrive\Área de Trabalho\Felipe\TI\Programação\PHYTON\Python2025\IA\bibliotecas\Pandas\athlete_events.csv')

# print(df.head())
# print(df.info())
# print(df.describe())

# 2️⃣ Limpeza
df.drop(columns=['ID'], inplace=True)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Height'] = df['Height'].fillna(df['Height'].mean())
df['Weight'] = df['Weight'].fillna(df['Weight'].mean())
df['Medal'] = df['Medal'].fillna('None')
df.drop_duplicates(inplace=True)

print(df.head())
print(df.columns)

# 3️⃣ Transformação categórica
ct = ColumnTransformer(
    transformers=[
        ('categoricas',OneHotEncoder(sparse_output=False),['Sex','Sport','Medal']),
        ('numericas',StandardScaler(),['Age','Weight','Height'])
        ],
)

x= ct.fit_transform(df)
print(x)







