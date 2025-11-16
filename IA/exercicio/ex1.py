import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

 
df = pd.DataFrame({
    'ID':[1,2,3],
    'Nome':['Rafa','Felipe','Thay'],
    'Sexo':['Male','Male','Female'],
    'Idade':[25,50,39],
    'Peso':[60,55,53],
    'Comprou': [0,1,1]
})

X = df.drop('Comprou',axis=1)
y = df['Comprou']

cat_cols= ['Sexo']
num_cols = ['Idade','Peso']

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(), cat_cols)
])

pipeline = Pipeline([
    ('prep', preprocess),
    ('model',GaussianNB())
])


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

print('Acurácia', accuracy_score(y_test,y_pred) )
