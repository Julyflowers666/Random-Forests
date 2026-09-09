import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Dados
dados = {
    "focos": [5, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    "temperatura": [28, 30, 31, 33, 35, 36, 38, 39, 40, 42],
    "umidade": [85, 80, 75, 65, 60, 50, 40, 35, 25, 20],
    "risco": [
        "baixo",
        "baixo",
        "baixo",
        "medio",
        "medio",
        "medio",
        "alto",
        "alto",
        "alto",
        "alto"
    ]
}

df = pd.DataFrame(dados)


# Entradas
X = df[["focos", "temperatura", "umidade"]]

# Resultado que queremos prever
y = df["risco"]


# Separando treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Criando o modelo
modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Treinando
modelo.fit(X_treino, y_treino)

# Testando
y_pred = modelo.predict(X_teste)

precisao = accuracy_score(y_teste, y_pred)

print("Precisão:", precisao)


# Nova previsão
#focos/temperatura/umidade
nova_regiao = [[55, 40, 25]]

previsao = modelo.predict(nova_regiao)

print("Risco previsto:", previsao[0])