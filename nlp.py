import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({
    "emocao":["alegria", "medo"]
})

coder = OneHotEncoder(sparse_output=False)
X = coder.fit_transform(df[["emocao"]])
print(X)
print(df.head())
print(coder.get_feature_names_out())