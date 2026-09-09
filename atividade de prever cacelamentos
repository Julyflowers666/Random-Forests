# ============================================================
# EXERCÍCIO: RANDOM FOREST PARA PREVER CANCELAMENTO DE CLIENTE
# ============================================================

# Importamos o pandas para trabalhar com os dados
import pandas as pd

# Função para separar os dados em treinamento e teste
from sklearn.model_selection import train_test_split

# Nosso algoritmo Random Forest
from sklearn.ensemble import RandomForestClassifier

# Métrica para verificar a quantidade de acertos
from sklearn.metrics import accuracy_score


# ============================================================
# 1. CRIANDO NOSSO DATASET
# ============================================================

# Vamos criar alguns clientes fictícios.
#
# idade            -> idade do cliente
# renda            -> renda mensal
# meses_cliente    -> há quantos meses é cliente
# cancelou         -> nossa variável que queremos prever
#
# 0 = não cancelou
# 1 = cancelou

dados = {
    "idade": [
        22, 45, 31, 50, 28,
        40, 35, 23, 52, 29,
        33, 48, 25, 55, 27,
        38, 21, 44, 30, 51
    ],

    "renda": [
        2000, 7000, 4000, 8000, 3000,
        6000, 5000, 2500, 9000, 3500,
        4500, 7500, 2800, 8500, 3200,
        5500, 1800, 6500, 4200, 8200
    ],

    "meses_cliente": [
        2, 36, 12, 48, 5,
        30, 20, 3, 60, 8,
        15, 42, 4, 55, 6,
        25, 1, 33, 10, 50
    ],

    "cancelou": [
        1, 0, 1, 0, 1,
        0, 0, 1, 0, 1,
        0, 0, 1, 0, 1,
        0, 1, 0, 1, 0
    ]
}


# Transformamos o dicionário em um DataFrame
df = pd.DataFrame(dados)


# Vamos visualizar os dados
print(df)


# ============================================================
# 2. SEPARANDO X E Y
# ============================================================

# X contém as informações que o modelo vai utilizar
# para tentar fazer a previsão.

X = df[
    [
        "idade",
        "renda",
        "meses_cliente"
    ]
]


# y é aquilo que queremos prever.
#
# Neste caso:
# 0 = não cancelou
# 1 = cancelou

y = df["cancelou"]


# ============================================================
# 3. SEPARANDO TREINAMENTO E TESTE
# ============================================================

# Vamos separar:
#
# 70% dos dados -> treinamento
# 30% dos dados -> teste
#
# O modelo aprende usando os dados de treinamento.
# Depois verificamos seu desempenho usando dados de teste.

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# ============================================================
# 4. CRIANDO O RANDOM FOREST
# ============================================================

modelo = RandomForestClassifier(
    n_estimators=100,  # quantidade de árvores
    random_state=42    # deixa o resultado reproduzível
)


# ============================================================
# 5. TREINANDO O MODELO
# ============================================================

# Aqui acontece o aprendizado.
#
# O Random Forest vai analisar os dados de treinamento
# e construir várias árvores de decisão.

modelo.fit(
    X_treino,
    y_treino
)


# ============================================================
# 6. FAZENDO PREVISÕES
# ============================================================

# Agora usamos os dados de teste para descobrir
# o que o modelo prevê.

previsoes = modelo.predict(X_teste)


# Vamos mostrar as previsões
print("\nPrevisões:")
print(previsoes)


# Vamos mostrar os valores reais
print("\nValores reais:")
print(y_teste.values)


# ============================================================
# 7. CALCULANDO A ACURÁCIA
# ============================================================

# A acurácia mostra a proporção de previsões corretas.

acuracia = accuracy_score(
    y_teste,
    previsoes
)

print("\nAcurácia:", acuracia)


# ============================================================
# 8. TESTANDO UM CLIENTE NOVO
# ============================================================

# Imagine que apareceu um novo cliente:
#
# idade = 27 anos
# renda = R$ 3.000
# meses como cliente = 6

novo_cliente = pd.DataFrame({
    "idade": [27],
    "renda": [3000],
    "meses_cliente": [6]
})


# Pedimos para o modelo fazer uma previsão
previsao = modelo.predict(novo_cliente)


# Mostramos o resultado
print("\nPrevisão para o novo cliente:")

if previsao[0] == 1:
    print("O cliente provavelmente vai CANCELAR.")
else:
    print("O cliente provavelmente NÃO vai cancelar.")


# ============================================================
# 9. DESCOBRINDO A PROBABILIDADE
# ============================================================

# predict_proba() mostra a probabilidade de cada classe.
#
# Coluna 0 -> probabilidade de NÃO cancelar
# Coluna 1 -> probabilidade de CANCELAR

probabilidade = modelo.predict_proba(novo_cliente)

print("\nProbabilidades:")
print(probabilidade)


# Podemos mostrar de uma forma mais amigável

print(
    f"Probabilidade de não cancelar: "
    f"{probabilidade[0][0] * 100:.2f}%"
)

print(
    f"Probabilidade de cancelar: "
    f"{probabilidade[0][1] * 100:.2f}%"
)
