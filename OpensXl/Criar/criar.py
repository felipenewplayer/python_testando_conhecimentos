import pandas as pd

alunos = {'Nome':['Ricador','Felipe','Thay','Roberto'],
          'Nota':[4,6,10,7],
          'Aprovado':['Não','Não','Sim','Sim']}

data_frame= pd.DataFrame(alunos)
print(data_frame.head())
print(f"\ntamanho de colunas e linhas {data_frame.shape}")
print(f"\n{data_frame.describe()}")
print(f"\nTabela filtrada pelo nome\n\n{data_frame['Nome']}")


print(f"\nFiltragem por indice\n\n {data_frame.loc[0:10]}")

alunos_aprovado = data_frame.loc[data_frame['Aprovado']=='Sim']
print(f"\Tabela de alunos aprovado\n{alunos_aprovado}")