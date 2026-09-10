import csv
import numpy as np

arquivo = "arsenio_dataset (1).csv"

X = []
y = []

with open(arquivo, "r") as f:
    leitor = csv.reader(f)
    next(leitor)

    for linha in leitor:
        X.append([
            float(linha[0]),
            float(linha[2]),
            float(linha[3]),
            float(linha[4]) 
        ])
        y.append(float(linha[5]))

X = np.array(X)
y = np.array(y)

X1 = np.column_stack((np.ones(len(X)), X))

b = np.linalg.inv(X1.T @ X1) @ X1.T @ y

y_prev = X1 @ b

novo_x = np.array([1, 30, 5, 5, 0.135])
previsao = novo_x @ b

print("Coeficientes:")
print("Intercepto =", b[0])
print("Idade =", b[1])
print("Beber =", b[2])
print("Cozinhar =", b[3])
print("Agua =", b[4])

print("\nPrevisão:")
print(previsao)

residuos = y - y_prev
ss_res = np.sum(residuos ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)

r2 = 1 - ss_res / ss_tot

print("\nR2:")
print(r2)

n = len(y)
p = 4

r2_ajustado = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("\nR2 ajustado:")
print(r2_ajustado)

X_agua = X[:, 3]
X_agua = np.column_stack((np.ones(len(X_agua)), X_agua))

b_agua = np.linalg.inv(X_agua.T @ X_agua) @ X_agua.T @ y
y_prev_agua = X_agua @ b_agua

res_agua = y - y_prev_agua

ss_res_agua = np.sum(res_agua ** 2)

r2_agua = 1 - ss_res_agua / ss_tot

print("\nR2 usando somente Agua:")
print(r2_agua)

b_sem = np.linalg.inv(X.T @ X) @ X.T @ y
y_prev_sem = X @ b_sem

res_sem = y - y_prev_sem

r2_sem = 1 - np.sum(res_sem ** 2) / ss_tot
rmse_sem = np.sqrt(np.mean(res_sem ** 2))

print("\nModelo sem intercepto:")
print("R2 =", r2_sem)
print("RMSE =", rmse_sem)

mse = np.mean(residuos ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(residuos))

mse_agua = np.mean(res_agua ** 2)
rmse_agua = np.sqrt(mse_agua)
mae_agua = np.mean(np.abs(res_agua))

print("\nMetricas do modelo completo:")
print("MSE =", mse)
print("RMSE =", rmse)
print("MAE =", mae)

print("\nMetricas do modelo com Agua:")
print("MSE =", mse_agua)
print("RMSE =", rmse_agua)
print("MAE =", mae_agua)