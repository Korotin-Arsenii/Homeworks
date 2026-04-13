import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Функція навантаження
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24

# 2. Точ    не значення інтегралу
I0, _ = quad(f, a, b, epsabs=1e-14)

# 3. Складова формула Сімпсона
def simpson(f, a, b, N):
    if N % 2 != 0: N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

# 4. Дослідження залежності точності від N
N_vals = np.arange(10, 1002, 2)
eps_vals = [abs(simpson(f, a, b, N) - I0) for N in N_vals]

# Пошук оптимального N
Nopt = next((N for N, eps in zip(N_vals, eps_vals) if eps <= 1e-12), N_vals[-1])
epsopt = abs(simpson(f, a, b, Nopt) - I0)

# 5. Похибка при N0
N0 = max(8, round(Nopt / 10 / 8) * 8)
I_N0 = simpson(f, a, b, N0)
eps0 = abs(I_N0 - I0)

# 6. Метод Рунге-Ромберга
I_N0_2 = simpson(f, a, b, N0 // 2)
I_R = I_N0 + (I_N0 - I_N0_2) / 15
epsR = abs(I_R - I0)

# 7. Метод Ейткена та порядок методу
I_N0_4 = simpson(f, a, b, N0 // 4)
I_E = (I_N0_2**2 - I_N0 * I_N0_4) / (2 * I_N0_2 - (I_N0 + I_N0_4))
p = (1 / np.log(2)) * np.log(abs((I_N0_4 - I_N0_2) / (I_N0_2 - I_N0)))
epsE = abs(I_E - I0)

# 9. Адаптивний алгоритм
def adaptive_simpson(f, a, b, tol):
    c = (a + b) / 2
    I1 = simpson(f, a, b, 2)
    I2 = simpson(f, a, c, 2) + simpson(f, c, b, 2)
    if abs(I1 - I2) <= 15 * tol:
        return I2 + (I2 - I1) / 15
    return adaptive_simpson(f, a, c, tol / 2) + adaptive_simpson(f, c, b, tol / 2)

tol_adapt = 1e-10
I_adapt = adaptive_simpson(f, a, b, tol_adapt)
eps_adapt = abs(I_adapt - I0)

# Вивід результатів
print("-" * 40)
print(f"1-2. Точне значення (I0): {I0:.10f}")
print(f"4. Nopt: {Nopt}, epsopt: {epsopt:.2e}")
print(f"5. N0: {N0}, eps0: {eps0:.2e}")
print(f"6. Рунге-Ромберг (I_R): {I_R:.10f}, epsR: {epsR:.2e}")
print(f"7. Ейткен (I_E): {I_E:.10f}, epsE: {epsE:.2e}, Порядок p: {p:.2f}")
print(f"9. Адаптивний (I_adapt): {I_adapt:.10f}, eps_adapt: {eps_adapt:.2e}")
print("-" * 40)

# 4. Побудова графіка
plt.figure(figsize=(10, 6))
plt.plot(N_vals, eps_vals, label=r'Похибка $\epsilon(N) = |I(N) - I_0|$', color='blue')
plt.axhline(1e-12, color='red', linestyle='--', label=r'Задана точність $1e-12$')
plt.axvline(Nopt, color='green', linestyle='--', label=f'$N_{{opt}} = {Nopt}$')
plt.yscale('log')
plt.title('Залежність точності обчислення від числа розбиття N')
plt.xlabel('Число розбиття, N')
plt.ylabel(r'Похибка, $\epsilon$')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()