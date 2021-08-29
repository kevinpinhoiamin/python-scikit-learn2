import pandas as pd
from collections import Counter
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('busca.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values



porcentagem_de_treino = 0.9
tamanho_de_treino = int(porcentagem_de_treino * len(X))

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados = X[tamanho_de_treino:]
teste_marcacoes = Y[tamanho_de_treino:]

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
acertos = resultado == teste_marcacoes

total_de_acertos = sum(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = total_de_acertos / total_de_elementos * 100

print(f'Taxa de acerto do algoritmo: {taxa_de_acerto}')
print(total_de_elementos)

# a eficacia do algoritmo que chuta tudo um unico valor
acerto_base = max(Counter(teste_marcacoes).values())
taxa_de_acerto_base = acerto_base / len(teste_marcacoes) * 100
print(f'Taxa de acerto base: {taxa_de_acerto_base}')
