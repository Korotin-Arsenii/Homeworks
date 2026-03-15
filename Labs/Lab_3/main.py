import csv
import matplotlib.pyplot as plt


# 1. Вхідні дані
def load_data(filename="data.csv"):
    x, y = [], []
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Пропуск заголовка
            for row in reader:
                x.append(float(row[0]))
                y.append(float(row[1]))
    except FileNotFoundError:
        # Дані за замовчуванням, якщо файлу немає
        x = list(range(1, 25))
        y = [-2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0, -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3]
    return x, y


x, y = load_data()
n_nodes = len(x)


# 2. Функції МНК
def form_matrix(x, m):
    A = [[0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            A[i][j] = sum(xi ** (i + j) for xi in x)
    return A


def form_vector(x, y, m):
    b = [0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(yi * (xi ** i) for xi, yi in zip(x, y))
    return b


def gauss_solve(A, b):
    n = len(b)
    A = [row[:] for row in A]
    b = list(b)

    for k in range(n - 1):
        max_row = k + max(range(n - k), key=lambda i: abs(A[k + i][k]))
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            if A[k][k] == 0: continue
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    x_sol = [0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x_sol[j] for j in range(i + 1, n))
        x_sol[i] = (b[i] - s) / A[i][i]
    return x_sol


def polynomial(x_vals, coef):
    return [sum(c * (xi ** i) for i, c in enumerate(coef)) for xi in x_vals]


def variance(y_true, y_approx):
    return sum((yt - ya) ** 2 for yt, ya in zip(y_true, y_approx)) / len(y_true)


# 3. Вибір оптимального ступеня полінома
max_degree = 10
variances = []
models = []

for m in range(1, max_degree + 1):
    A = form_matrix(x, m)
    b_vec = form_vector(x, y, m)
    try:
        coef = gauss_solve(A, b_vec)
        y_approx = polynomial(x, coef)
        var = variance(y, y_approx)
    except ZeroDivisionError:
        var = float('inf')
        coef = []

    variances.append(var)
    models.append(coef)

optimal_m = variances.index(min(variances)) + 1
optimal_coef = models[optimal_m - 1]

# 4. Побудова апроксимації
y_approx_opt = polynomial(x, optimal_coef)

# 5. Прогноз на наступні 3 місяці
x_future = [x[-1] + 1, x[-1] + 2, x[-1] + 3]
y_future = polynomial(x_future, optimal_coef)

# 6. Похибка апроксимації
h1 = (x[-1] - x[0]) / (20 * n_nodes)
x_tab = [x[0] + i * h1 for i in range(int((x[-1] - x[0]) / h1) + 1)]
y_tab_approx = polynomial(x_tab, optimal_coef)

# Розрахунок похибки (екстраполяція лінійно для точок між вузлами)
error_y = [abs(yi - ya) for yi, ya in zip(y, y_approx_opt)]

# 7. Вивід результатів
print("=== Дисперсії для різних ступенів m ===")
for m, var in enumerate(variances, 1):
    print(f"m = {m:2d} | Дисперсія = {var:.4f}")

print(f"\nОптимальний ступінь m = {optimal_m}")

print("\n=== Прогноз на наступні 3 місяці ===")
for x_f, y_f in zip(x_future, y_future):
    print(f"Місяць {int(x_f)}: {y_f:.2f}")

# Побудова графіків
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Графік 1: Фактичні дані та апроксимація
axs[0].scatter(x, y, color='red', label='Фактичні дані')
axs[0].plot(x, y_approx_opt, color='blue', label=f'Апроксимація (m={optimal_m})')
axs[0].set_title('Дані та найкраще квадратичне наближення')
axs[0].legend()
axs[0].grid(True)

# Графік 2: Похибка апроксимації
axs[1].plot(x, error_y, color='orange', marker='o', label='Похибка у вузлах')
axs[1].set_title('Графік похибки апроксимації')
axs[1].legend()
axs[1].grid(True)

# Графік 3: Дисперсія від степеня
axs[2].plot(range(1, max_degree + 1), variances, marker='s', color='green')
axs[2].set_title('Залежність дисперсії від степеня многочлена')
axs[2].set_xlabel('Ступінь m')
axs[2].set_ylabel('Дисперсія')
axs[2].grid(True)

plt.tight_layout()
plt.show()