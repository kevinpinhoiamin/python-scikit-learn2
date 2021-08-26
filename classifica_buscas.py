import pandas as pd
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
diferenca = resultado - teste_marcacoes

acertos = [d for d in diferenca if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = total_de_acertos / total_de_elementos * 100

print(taxa_de_acerto)
print(total_de_elementos)
