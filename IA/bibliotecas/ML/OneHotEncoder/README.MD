# OneHotEncoder

Cria colunas binárias (0 ou 1) para cada categoria.

| Categoria | Cachorro | Gato | Coelho |
| --------- | -------- | ---- | ------ |
| Gato      | 0        | 1    | 0      |
| Cachorro  | 1        | 0    | 0      |
| Coelho    | 0        | 0    | 1      |

from sklearn.preprocessing import OneHotEncoder
import numpy as np

animais = np.array(['Gato', 'Cachorro', 'Coelho', 'Cachorro']).reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
valores = encoder.fit_transform(animais)

print(valores)
# Saída:
# [[0. 1. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
print(encoder.get_feature_names_out())
# ['x0_Cachorro' 'x0_Gato' 'x0_Coelho']

