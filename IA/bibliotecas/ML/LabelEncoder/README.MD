# LabelEncoder

## Server para transformar dados categóricos em dados númericos, já que os modelos matemáticos não entendem texto diretamente


Transforma categorias em números inteiros.

| Categoria | Código |
| --------- | ------ |
| Gato      | 0      |
| Cachorro  | 1      |
| Coelho    | 2      |


from sklearn.preprocessing import LabelEncoder

animais = ['Gato', 'Cachorro', 'Coelho', 'Cachorro', 'Gato']

encoder = LabelEncoder()
valores = encoder.fit_transform(animais)

print(valores)  
# Saída: [1 0 2 0 1]
print(encoder.classes_)  
# Saída: ['Cachorro', 'Gato', 'Coelho']


Cada nome é convertido em um número inteiro único.

Mas cuidado: o modelo pode interpretar isso como ordem numérica, o que nem sempre faz sentido.



