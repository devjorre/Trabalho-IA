import csv
import numpy as np

arquivo = "dose_radiacao_expandido.csv"

X = []
y = []

with open(arquivo, "r") as f:
    leitor = csv.reader(f)
    next(leitor)

    for linha in leitor:
        X.append([float(linha[2]), float(linha[3])])
        y.append(float(linha[1]))

X = np.array(X)
y = np.array(y)

X1 = np.column_stack((np.ones(len(X)), X))

b = np.linalg.inv(X1.T @ X1) @ X1.T @ y

y_prev = X1 @ b

novo_x = np.array([1, 15, 5])
previsao = novo_x @ b

print("Coeficientes:")
print(b)

print("\nPrevisão para 15 mAmp e 5 min:")
print(previsao)

residuos = y - y_prev
ss_res = np.sum(residuos ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)

r2 = 1 - ss_res / ss_tot

print("\nR2:")
print(r2)

n = len(y)
p = 2

r2_ajustado = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("\nR2 ajustado:")
print(r2_ajustado)

mse = np.mean(residuos ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(residuos))

print("\nMétricas:")
print("MSE =", mse)
print("RMSE =", rmse)
print("MAE =", mae)