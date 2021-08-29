import pandas as pd
from collections import Counter

# teste inicial: home, buscas, logado => comprou
# home, busca
# home, logado
# busca, logado
# busca: 75% (7 testes)

df = pd.read_csv('busca.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_teste = int(porcentagem_de_teste * len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

fim_de_teste = tamanho_de_treino + tamanho_de_teste
teste_dados = X[tamanho_de_treino:fim_de_teste]
teste_marcacoes = Y[tamanho_de_treino:fim_de_teste]

validacao_dados = X[fim_de_teste:]
validacao_marcacoes = Y[fim_de_teste:]


def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)
    acertos = resultado == teste_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)
    taxa_de_acerto = total_de_acertos / total_de_elementos * 100

    print(f'Taxa de acerto do {nome}: {taxa_de_acerto}')

    return taxa_de_acerto


from sklearn.naive_bayes import MultinomialNB

modelo_multinomial = MultinomialNB()
resultado_multinomial = fit_and_predict('MultinomialNB', modelo_multinomial, treino_dados, treino_marcacoes,
                                        teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier

modelo_ada_boost = AdaBoostClassifier()
resultado_ada_boost = fit_and_predict('AdaBoostClassifier', modelo_ada_boost, treino_dados, treino_marcacoes,
                                      teste_dados, teste_marcacoes)



vencedor = None
if resultado_multinomial > resultado_ada_boost:
    vencedor = modelo_multinomial
else:
    vencedor = modelo_ada_boost

resultado = vencedor.predict(validacao_dados)
acertos = resultado == validacao_marcacoes

total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_dados)
taxa_de_acerto = total_de_acertos / total_de_elementos * 100

print(f'Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {taxa_de_acerto}')



# a eficacia do algoritmo que chuta tudo um unico valor
acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = acerto_base / len(validacao_marcacoes) * 100
print(f'Taxa de acerto base: {taxa_de_acerto_base}')

print(f'Total de testes: {len(validacao_dados)}')
