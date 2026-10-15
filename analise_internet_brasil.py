
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn import linear_model

# Pasta onde os graficos serao salvos
PASTA_GRAFICOS = Path("graficos")
PASTA_GRAFICOS.mkdir(exist_ok=True)

# Dados de uso da internet no Brasil
anos = list(range(2000, 2025))

uso = [
    2.87, 4.53, 9.15, 13.21, 19.07,
    21.02, 28.18, 30.88, 33.83, 39.22,
    40.65, 45.69, 48.56, 51.04, 54.55,
    58.33, 60.87, 67.47, 70.43, 73.91,
    81.34, 80.69, 80.53, 84.15, 84.46
]

base = pd.DataFrame({
    "Ano": anos,
    "Uso": uso
})

base["x"] = base["Ano"] - 2000

print("Dados utilizados:")
print(base.head())

print("\nEstatistica descritiva:")
print(base[["x", "Uso"]].describe().round(2))

# Grafico 1: diagrama de dispersao
plt.figure(figsize=(9, 5))
plt.scatter(base["Ano"], base["Uso"], color="gray")
plt.title("Uso de internet no Brasil (2000 a 2024)")
plt.xlabel("Ano")
plt.ylabel("% da populacao que usa internet")
plt.xticks([2000, 2004, 2008, 2012, 2016, 2020, 2024])
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "grafico_dispersao.png", dpi=150)
plt.show()

x = list(base["x"])
y = list(base["Uso"])
n = len(x)

# Calculo dos somatorios
soma_x = 0
soma_y = 0
soma_x2 = 0
soma_y2 = 0
soma_xy = 0

for i in range(n):
    soma_x += x[i]
    soma_y += y[i]
    soma_x2 += x[i] ** 2
    soma_y2 += y[i] ** 2
    soma_xy += x[i] * y[i]

print("\nSomatorios:")
print("n =", n)
print("soma de x =", soma_x)
print("soma de y =", round(soma_y, 2))
print("soma de x2 =", soma_x2)
print("soma de y2 =", round(soma_y2, 4))
print("soma de xy =", round(soma_xy, 2))

# Correlacao de Pearson
cima = n * soma_xy - soma_x * soma_y
baixo_x = n * soma_x2 - soma_x ** 2
baixo_y = n * soma_y2 - soma_y ** 2

r = cima / math.sqrt(baixo_x * baixo_y)
r2 = r ** 2

print("\nCorrelacao e determinacao:")
print("r =", round(r, 4))
print("r2 =", round(r2, 4))

# Teste de significancia
t = r * math.sqrt((n - 2) / (1 - r2))
p = 2 * stats.t.sf(abs(t), n - 2)

print("\nTeste de significancia:")
print("t =", round(t, 2))
print("valor-p = %.2e" % p)

# Equacao da reta de regressao
b = cima / baixo_x
a = soma_y / n - b * soma_x / n

print("\nReta de regressao:")
print("b =", round(b, 4))
print("a =", round(a, 4))
print("reta: y = %.2f + %.2f x" % (a, b))

# Calculo dos residuos
erros = []
soma_erros2 = 0

for i in range(n):
    valor_reta = a + b * x[i]
    erro = y[i] - valor_reta
    erros.append(erro)
    soma_erros2 += erro ** 2

erro_padrao = math.sqrt(soma_erros2 / (n - 2))
sxx = soma_x2 - soma_x ** 2 / n
erro_b = erro_padrao / math.sqrt(sxx)
t_critico = stats.t.ppf(0.975, n - 2)

print("Erro padrao da estimativa =", round(erro_padrao, 2))
print("Erro padrao de b =", round(erro_b, 3))
print(
    "Intervalo de 95% para b: de %.2f ate %.2f"
    % (
        b - t_critico * erro_b,
        b + t_critico * erro_b
    )
)

# Grafico 2: reta de regressao
reta = [a + b * valor for valor in x]

plt.figure(figsize=(9, 5))
plt.scatter(base["Ano"], base["Uso"], color="gray", label="Dados")
plt.plot(
    base["Ano"],
    reta,
    color="blue",
    linewidth=2,
    label="Reta de regressao"
)
plt.title("Diagrama de dispersao e reta de regressao")
plt.xlabel("Ano")
plt.ylabel("% da populacao que usa internet")
plt.xticks([2000, 2004, 2008, 2012, 2016, 2020, 2024])
plt.ylim(0, 100)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "grafico_reta.png", dpi=150)
plt.show()

# Grafico 3: residuos
plt.figure(figsize=(9, 4))
plt.bar(base["Ano"], erros, color="gray")
plt.axhline(0, color="black", linewidth=1)
plt.title("Residuos da reta (valor real - valor previsto)")
plt.xlabel("Ano")
plt.ylabel("Pontos percentuais")
plt.xticks([2000, 2004, 2008, 2012, 2016, 2020, 2024])
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "grafico_residuos.png", dpi=150)
plt.show()

# Conferencia com scikit-learn
X = base[["x"]].values
Y = base["Uso"].values

reg = linear_model.LinearRegression()
reg.fit(X, Y)

print("\nConferindo com as bibliotecas:")
print(base[["x", "Uso"]].corr().round(4))
print("b (sklearn) =", round(reg.coef_[0], 4))
print("a (sklearn) =", round(reg.intercept_, 4))
print("r2 (sklearn) =", round(reg.score(X, Y), 4))

# Previsoes futuras
print("\nPrevisoes:")

for ano in [2025, 2026, 2027, 2030]:
    xn = ano - 2000
    manual = a + b * xn
    com_sklearn = reg.predict([[xn]])[0]

    print(
        ano,
        "->",
        round(manual, 2),
        "% (sklearn:",
        round(com_sklearn, 2),
        "%)"
    )

ano_100 = 2000 + (100 - a) / b
print("A reta chega em 100% por volta de", round(ano_100, 1))

# Teste com dados de 2000 a 2019 e previsoes de 2020 a 2024
treino = base[base["Ano"] <= 2019]
teste = base[base["Ano"] >= 2020]

reg2 = linear_model.LinearRegression()
reg2.fit(treino[["x"]].values, treino["Uso"].values)

prev_teste = reg2.predict(teste[["x"]].values)

print("\nTeste do modelo com anos nao utilizados no ajuste:")
print(
    "Modelo treinado ate 2019: y = %.4f + %.4f x"
    % (reg2.intercept_, reg2.coef_[0])
)

soma_abs = 0

for j, (_, linha) in enumerate(teste.iterrows()):
    ano = int(linha["Ano"])
    real = linha["Uso"]
    previsto = prev_teste[j]
    erro = real - previsto

    soma_abs += abs(erro)

    print(
        ano,
        "real:", round(real, 2),
        "previsto:", round(previsto, 2),
        "erro:", round(erro, 2)
    )

print("Erro medio absoluto =", round(soma_abs / len(teste), 2))
print("\nGraficos salvos em:", PASTA_GRAFICOS.resolve())
