import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# ПУНКТ 1: АНАЛІТИЧНИЙ РОЗВ'ЯЗОК ТА ВХІДНІ ДАНІ
# ==============================================================================
# Диференціальне рівняння: dy/dx = x + y
def f(x, y):
    return x + y


# Точний (аналітичний) розв'язок рівняння для початкової умови y(0) = 1
def y_exact(x):
    return 2 * np.exp(x) - x - 1


a, b = 0.0, 1.0  # Відрізок інтегрування
y0 = 1.0  # Початкова умова y(a) = y0
h_fixed = 10 ** -2  # Фіксований крок
eps = 10 ** -5  # Задана точність для автоматичного кроку

# ==============================================================================
# ЧАСТИНА 1 (АДАМС)
# ==============================================================================

# ПУНКТ 2: ЧИСЕЛЬНИЙ РОЗВ'ЯЗОК МЕТОДОМ АДАМСА 2-ГО ПОРЯДКУ (ФІКСОВАНИЙ КРОК)
x_ad = np.arange(a, b + h_fixed, h_fixed)
y_ad = np.zeros(len(x_ad))
y_np_arr = np.zeros(len(x_ad))
y_kop_arr = np.zeros(len(x_ad))

y_ad[0] = y0
# Перший крок знаходимо за допомогою РК4
k1 = f(x_ad[0], y_ad[0])
k2 = f(x_ad[0] + h_fixed / 2, y_ad[0] + h_fixed * k1 / 2)
k3 = f(x_ad[0] + h_fixed / 2, y_ad[0] + h_fixed * k2 / 2)
k4 = f(x_ad[0] + h_fixed, y_ad[0] + h_fixed * k3)
y_ad[1] = y_ad[0] + (h_fixed / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

for n in range(1, len(x_ad) - 1):
    f_n = f(x_ad[n], y_ad[n])
    f_nm1 = f(x_ad[n - 1], y_ad[n - 1])

    # Прогноз (Предиктор)
    y_np = y_ad[n] + (h_fixed / 2) * (3 * f_n - f_nm1)
    y_np_arr[n + 1] = y_np

    # Модифікація
    y_mod = y_np + (5 / 6) * (y_kop_arr[n] - y_np_arr[n]) if n > 1 else y_np

    # Корекція (Коректор — 2 ітерації)
    y_kop = y_mod
    for _ in range(2):
        y_kop = y_ad[n] + (h_fixed / 2) * (f(x_ad[n + 1], y_kop) + f_n)

    y_kop_arr[n + 1] = y_kop
    y_ad[n + 1] = y_kop - (1 / 6) * (y_kop - y_np)

# ПУНКТ 3: ОБЧИСЛЕННЯ ТОЧНОЇ ЛОКАЛЬНОЇ ПОХИБКИ АДАМСА
err_exact_ad = y_ad - y_exact(x_ad)

# ПУНКТ 4: ОБЧИСЛЕННЯ ПОХИБКИ ЗА ВИРАЗОМ (y_kop - y_np)
err_est_ad = np.zeros(len(x_ad))
for n in range(1, len(x_ad) - 1):
    err_est_ad[n + 1] = abs(y_kop_arr[n + 1] - y_np_arr[n + 1])


# ПУНКТ 5: АВТОМАТИЧНИЙ ВИБІР КРОКУ ДЛЯ МЕТОДУ АДАМСА
def adams_auto(a, b, y0, eps):
    x_arr, h_arr = [a, a + h_fixed], [h_fixed]
    # Старт через РК4
    k1 = f(a, y0)
    k2 = f(a + h_fixed / 2, y0 + h_fixed * k1 / 2)
    k3 = f(a + h_fixed / 2, y0 + h_fixed * k2 / 2)
    k4 = f(a + h_fixed, y0 + h_fixed * k3)
    y1 = y0 + (h_fixed / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    y_arr = [y0, y1]

    x = a + h_fixed
    h = h_fixed

    while x < b:
        if x + h > b: h = b - x
        n = len(y_arr) - 1

        f_n = f(x_arr[n], y_arr[n])
        f_nm1 = f(x_arr[n - 1], y_arr[n - 1])

        y_np = y_arr[n] + (h / 2) * (3 * f_n - f_nm1)
        y_kop = y_arr[n] + (h / 2) * (f(x + h, y_np) + f_n)

        err = (1 / 6) * abs(y_kop - y_np)

        if err > eps:
            h /= 2
            x_arr[-1] = x_arr[-2] + h
            # Перерахунок проміжної точки через РК4
            k1 = f(x_arr[-2], y_arr[-2])
            k2 = f(x_arr[-2] + h / 2, y_arr[-2] + h * k1 / 2)
            k3 = f(x_arr[-2] + h / 2, y_arr[-2] + h * k2 / 2)
            k4 = f(x_arr[-2] + h, y_arr[-2] + h * k3)
            y_arr[-1] = y_arr[-2] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            x = x_arr[-1]
        else:
            x += h
            y_final = y_kop - (1 / 6) * (y_kop - y_np)
            x_arr.append(x)
            y_arr.append(y_final)
            h_arr.append(h)
            if err < eps / 8:
                h *= 2

    return np.array(x_arr[:-1]), np.array(h_arr)


x_ad_auto, h_ad_auto = adams_auto(a, b, y0, eps)


# ==============================================================================
# ЧАСТИНА 2 (РУНГЕ-КУТТА)
# ==============================================================================

# Допоміжна функція для одного кроку РК4
def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# ПУНКТ 6: ЧИСЕЛЬНИЙ РОЗВ'ЯЗОК МЕТОДОМ РУНГЕ-КУТТА 4-ГО ПОРЯДКУ (h = 10^-2)
x_rk = np.arange(a, b + h_fixed, h_fixed)
y_rk = np.zeros(len(x_rk))
y_rk[0] = y0

for i in range(len(x_rk) - 1):
    y_rk[i + 1] = rk4_step(x_rk[i], y_rk[i], h_fixed)

# ПУНКТ 7: ОБЧИСЛЕННЯ ЛОКАЛЬНОЇ ПОХИБКИ РК4 ТА ДОСЛІДЖЕННЯ ЗАЛЕЖНОСТІ ВІД КРОКУ
err_exact_rk = y_rk - y_exact(x_rk)

# Дослідження залежності похибки РК4 від кроку (тест для h = 0.1 та h = 0.01)
x_h1 = np.arange(a, b + 0.1, 0.1)
y_rk_h1 = np.zeros(len(x_h1))
y_rk_h1[0] = y0
for i in range(len(x_h1) - 1):
    y_rk_h1[i + 1] = rk4_step(x_h1[i], y_rk_h1[i], 0.1)
err_rk_h1 = abs(y_rk_h1[-1] - y_exact(b))
err_rk_h2 = abs(y_rk[-1] - y_exact(b))

print("=" * 60)
print("ДОСЛІДЖЕННЯ ПОХИБКИ ВІД ВЕЛИЧИНИ КРОКУ (Пункт 7):")
print(f"Глобальна похибка РК4 при h = 0.1:  {err_rk_h1:.2e}")
print(f"Глобальна похибка РК4 при h = 0.01: {err_rk_h2:.2e}")
print(f"Фактичне зменшення похибки: {err_rk_h1 / err_rk_h2:.2f} разів (теоретично: 10000)")
print("-" * 60)

# ПУНКТ 8: ОБЧИСЛЕННЯ ПОХИБКИ ЗА МЕТОДОМ РУНГЕ ТА ОЦІНКА НЕОБХІДНОГО КРОКУ
err_runge = np.zeros(len(x_rk))
for i in range(len(x_rk)):
    yh = rk4_step(x_rk[i], y_rk[i], h_fixed)
    yh2_1 = rk4_step(x_rk[i], y_rk[i], h_fixed / 2)
    yh2_2 = rk4_step(x_rk[i] + h_fixed / 2, yh2_1, h_fixed / 2)
    err_runge[i] = (16 / 15) * abs(yh2_2 - yh)

# Теоретичний оптимальний крок для точності eps
h_optimal_rk = 0.1 * (eps / err_rk_h1) ** (1 / 4)
print("ОЦІНКА ОПТИМАЛЬНОГО КРОКУ ЗА МЕТОДОМ РУНГЕ (Пункт 8):")
print(f"Необхідний теоретичний крок для точності eps={eps}: h <= {h_optimal_rk:.4f}")
print("Висновок: Крок h = 0.01 є оптимальним із значним запасом точності.")
print("=" * 60)


# ПУНКТ 9: АВТОМАТИЧНИЙ ВИБІР КРОКУ ДЛЯ МЕТОДУ РУНГЕ-КУТТА
def rk4_auto(a, b, y0, eps):
    x_arr, h_arr = [a], []
    x, y, h = a, y0, h_fixed
    k_const = 32  # 2^(4+1)

    while x < b:
        if x + h > b: h = b - x

        yh = rk4_step(x, y, h)
        yh2_1 = rk4_step(x, y, h / 2)
        yh2_2 = rk4_step(x + h / 2, yh2_1, h / 2)

        err = (16 / 15) * abs(yh2_2 - yh)

        if err > eps:
            h /= 2
        else:
            x += h
            y = yh
            x_arr.append(x)
            h_arr.append(h)
            if err < eps / k_const:
                h *= 2

    return np.array(x_arr[:-1]), np.array(h_arr)


x_rk_auto, h_rk_auto = rk4_auto(a, b, y0, eps)

# ==============================================================================
# ПУНКТ 10: ПОБУДОВА ГРАФІКІВ ДЛЯ ОБОХ ЧАСТИН ЗВІТУ
# ==============================================================================
plt.figure(figsize=(12, 10))

# Графік 1 (Пункт 3, 4): Похибки Адамса
plt.subplot(2, 2, 1)
plt.plot(x_ad[2:], err_exact_ad[2:], label='Точна похибка')
plt.plot(x_ad[2:], err_est_ad[2:], '--', label='Оцінка (y_kop - y_np)')
plt.title('Похибки Метод Адамса 2-го порядку')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

# Графік 2 (Пункт 5): Автоматичний крок Адамса
plt.subplot(2, 2, 2)
plt.plot(x_ad_auto, h_ad_auto, 'r-', label='h(x) Адамс')
plt.title('Залежність кроку h(x) від x (Адамс)')
plt.xlabel('x')
plt.ylabel('Величина кроку h')
plt.legend()
plt.grid(True)

# Графік 3 (Пункт 7, 8): Похибки Рунге-Кутта
plt.subplot(2, 2, 3)
plt.plot(x_rk, err_exact_rk, label='Точна похибка (RK4)')
plt.plot(x_rk, err_runge, '--', label='Похибка за Рунге')
plt.title('Похибки Метод Рунге-Кутта 4-го порядку')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

# Графік 4 (Пункт 9): Автоматичний крок Рунге-Кутта
plt.subplot(2, 2, 4)
plt.plot(x_rk_auto, h_rk_auto, 'g-', label='h(x) RK4')
plt.title('Залежність кроку h(x) від x (RK4)')
plt.xlabel('x')
plt.ylabel('Величина кроку h')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n[Пункт 10]: Розрахунки завершено. Дані та графіки готові для звіту.")

