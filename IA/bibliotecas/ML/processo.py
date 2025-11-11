import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

# 3️⃣ Transformação categórica
cat_cols = ['Sex', 'Team', 'NOC', 'Games', 'Season', 'City', 'Sport', 'Event', 'Medal']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

# 4️⃣ Concatenar numéricas + codificadas
numeric_df = df.drop(columns=cat_cols + ['Name'])
df_final = pd.concat([numeric_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# print(df_final.shape)
# print(df_final.head())

# 5️⃣ Escalonar
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_final)

# Converter pra DataFrame (pra visualizar)
df_scaled = pd.DataFrame(df_scaled, columns=df_final.columns)
print(df_scaled[:5])
