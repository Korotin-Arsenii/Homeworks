import csv
import matplotlib.pyplot as plt


# --- 1. Зчитування даних ---
def read_data(filename):
    x, y = [], []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return x, y


x_exp, y_exp = read_data("data.csv")


# --- Базові математичні функції ---
def divided_diff(x, y):
    n = len(y)
    table = [[0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])
    return [table[0][i] for i in range(n)]


def newton_poly(coef, x_data, x_val):
    n = len(coef)
    res = coef[n - 1]
    for i in range(n - 2, -1, -1):
        res = res * (x_val - x_data[i]) + coef[i]
    return res


coef_exp = divided_diff(x_exp, y_exp)


def exact_f(x_val):
    return newton_poly(coef_exp, x_exp, x_val)


# --- 2. Таблиця розділених різниць ---
print("--- 2. Таблиця розділених різниць (коефіцієнти) ---")
print([round(c, 8) for c in coef_exp])
print()

# --- 3. Обчислення P(6000) ---
p_6000 = newton_poly(coef_exp, x_exp, 6000)
print("--- 3. Прогноз ---")
print(f"P(6000) Ньютон: {p_6000:.4f} мс\n")

# --- 5 і 6. Повторення для 5, 10, 20 вузлів та аналіз похибки ---
print("--- 5-6. Аналіз похибки (5, 10, 20 вузлів) ---")
x_dense = [1000 + i * 100 for i in range(151)]

for n_nodes in [5, 10, 20]:
    step = (16000 - 1000) / (n_nodes - 1)
    x_nodes = [1000 + i * step for i in range(n_nodes)]
    y_nodes = [exact_f(xi) for xi in x_nodes]
    coef_n = divided_diff(x_nodes, y_nodes)

    max_err = 0
    for xp in x_dense:
        err = abs(exact_f(xp) - newton_poly(coef_n, x_nodes, xp))
        if err > max_err:
            max_err = err
    print(f"Вузлів: {n_nodes} | Макс. похибка: {max_err:.6e}")
print()

# --- Дослідницька частина 1: Вплив кроку ---
print("--- Дослідження 1: Вплив кроку (Фіксований інтервал [1000, 16000]) ---")
for n in [5, 10, 15]:
    step = (16000 - 1000) / (n - 1)
    x_n = [1000 + i * step for i in range(n)]
    y_n = [exact_f(xn) for xn in x_n]
    coef_n = divided_diff(x_n, y_n)

    max_err = 0
    for xp in x_dense:
        err = abs(exact_f(xp) - newton_poly(coef_n, x_n, xp))
        if err > max_err:
            max_err = err
    print(f"Вузлів: {n} | Крок: {step:.2f} | Макс. похибка: {max_err:.6e}")
print()

# --- Дослідницька частина 2: Вплив кількості вузлів ---
print("--- Дослідження 2: Вплив кількості вузлів (Фіксований крок h=1000) ---")
h = 1000
for n in [5, 10, 15]:
    x_n = [1000 + i * h for i in range(n)]
    y_n = [exact_f(xn) for xn in x_n]
    coef_n = divided_diff(x_n, y_n)

    x_check = [1000 + i * 100 for i in range(int((x_n[-1] - 1000) / 100) + 1)]
    max_err = 0
    for xp in x_check:
        err = abs(exact_f(xp) - newton_poly(coef_n, x_n, xp))
        if err > max_err:
            max_err = err
    print(f"Вузлів: {n} | Інтервал: [1000, {x_n[-1]}] | Макс. похибка: {max_err:.6e}")

# --- 4. Побудова графіків ---
plt.figure(figsize=(10, 6))
y_dense = [exact_f(xi) for xi in x_dense]

plt.plot(x_exp, y_exp, 'ro', markersize=8, label='Експериментальні точки')
plt.plot(x_dense, y_dense, 'b-', linewidth=2, label='Інтерполяційна крива')
plt.plot(6000, p_6000, 'g*', markersize=12, label=f'P(6000) = {p_6000:.2f}')

plt.xlabel('n (розмір вхідних даних)')
plt.ylabel('t (мс)')
plt.title('Інтерполяція Ньютона')
plt.legend()
plt.grid(True)
plt.show()