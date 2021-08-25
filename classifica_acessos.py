# minha abordagem inicial foi
# 1. separar 90% para treino e 10% para teste: 88,89% de acerto.

from dados import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

X, Y = carregar_acessos()

treino_dados = X[:90]
treino_marcacoes = Y[:90]

teste_dados = X[90:]
teste_marcacoes = Y[90:]

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
