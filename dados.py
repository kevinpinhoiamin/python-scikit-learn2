import csv


def carregar_acessos():
    X = []
    Y = []

    arquivo = open('acesso.csv', 'r')
    leitor = csv.reader(arquivo)
    leitor.__next__()

    for home, como_funciona, contato, comprou in leitor:
        dados = [int(home), int(como_funciona), int(contato)]
        X.append(dados)
        Y.append(int(comprou))

    return X, Y

def carregar_buscas():
    X = []
    Y = []

    arquivo = open('busca.csv', 'r')
    leitor = csv.reader(arquivo)
    leitor.__next__()

    for home, busca, logado, comprou in leitor:
        dados = [int(home), busca, int(logado)]
        X.append(dados)
        Y.append(int(comprou))

    return X, Y
