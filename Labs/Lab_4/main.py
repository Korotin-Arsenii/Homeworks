import numpy as np
import matplotlib.pyplot as plt

# 1. Аналітичний вираз та точне значення
def f(x): return 50 * np.exp(-0.1 * x) + 5 * np.sin(x)
def df_exact(x): return -5 * np.exp(-0.1 * x) + 5 * np.cos(x)

x0 = 1.0
exact_val = df_exact(x0)

# 2. Дослідження залежності похибки від кроку h
h_vals = np.logspace(-20, 3, 1000)
# Уникнення ділення на нуль та помилок точності
with np.errstate(divide='ignore', invalid='ignore'):
    df_approx = (f(x0 + h_vals) - f(x0 - h_vals)) / (2 * h_vals)
    R = np.abs(df_approx - exact_val)

valid_idx = np.where(~np.isnan(R))[0]
best_idx = valid_idx[np.nanargmin(R[valid_idx])]
h0 = h_vals[best_idx]
R0 = R[best_idx]

# Побудова графіка
plt.figure(figsize=(8, 5))
plt.loglog(h_vals, R)
plt.xlabel('Крок h')
plt.ylabel('Похибка R')
plt.title('Залежність похибки чисельного диференціювання від кроку h')
plt.grid(True, which="both", ls="--")
plt.axvline(h0, color='r', linestyle='--', label=f'Оптимальне h0 = {h0:.1e}')
plt.legend()
plt.show()

# 3. Приймаємо значення кроку h
h = 10**-3

# 4. Обчислення за двома кроками
df_h = (f(x0 + h) - f(x0 - h)) / (2 * h)
df_2h = (f(x0 + 2*h) - f(x0 - 2*h)) / (4 * h)

# 5. Похибка R1
R1 = np.abs(df_h - exact_val)

# 6. Метод Рунге-Ромберга
df_R = df_h + (df_h - df_2h) / 3
R2 = np.abs(df_R - exact_val)

# 7. Метод Ейткена та три кроки
df_4h = (f(x0 + 4*h) - f(x0 - 4*h)) / (8 * h)

df_E = (df_2h**2 - df_4h * df_h) / (2 * df_2h - (df_4h + df_h))
p = (1 / np.log(2)) * np.log(np.abs((df_4h - df_2h) / (df_2h - df_h)))
R3 = np.abs(df_E - exact_val)

# Вивід фінальних результатів
print(f"Точне значення: {exact_val:.6f}")
print(f"Оптимальний крок h0: {h0:.1e}, похибка R0: {R0:.1e}")
print(f"Крок h = 10^-3, похибка R1: {R1:.1e}")
print(f"Рунге-Ромберг: {df_R:.6f}, похибка R2: {R2:.1e}")
print(f"Ейткен: {df_E:.6f}, похибка R3: {R3:.1e}")
print(f"Порядок точності (Ейткен) p: {p:.2f}")