import math
import cmath
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# ТРАНСЦЕНДЕНТНА ЧАСТИНА (Пункти 1-4)
# =============================================================================

# Вихідне рівняння F(x) = 0
def F(x): return x ** 2 - math.cos(x)


def dF(x): return 2 * x + math.sin(x)


def d2F(x): return 2 + math.cos(x)

        
# Критерій знаходження кореня (Пункт 3)
def check_stop(x_new, x_old, eps):
    return abs(F(x_new)) < eps and abs(x_new - x_old) < eps


def point_1_tabulation(a, b, h):
    """Табуляція, запис у файл та пошук наближень (Пункт 1)"""
    nodes = []
    x = a
    with open("tabulation.txt", "w") as f:
        while x <= b:
            y = F(x)
            f.write(f"{x:.4f}\t{y:.4f}\n")
            nodes.append((x, y))
            x += h

    roots_approx = []
    print("--- Пункт 1: Аналіз точок перетину ---")
    for i in range(len(nodes) - 1):
        if nodes[i][1] * nodes[i + 1][1] < 0:
            avg_x = (nodes[i][0] + nodes[i + 1][0]) / 2
            behavior = "зростання" if nodes[i + 1][1] > nodes[i][1] else "спадання"
            roots_approx.append(avg_x)
            print(f"Знайдено корінь: x ≈ {avg_x:.2f} ({behavior})")
    return roots_approx


# Ітераційні методи (Пункт 2)
def simple_iteration(x0, eps=1e-10):
    tau = -0.5 if x0 > 0 else 0.5
    iters, x = 0, x0
    while iters < 1000:
        x_new = x + tau * F(x)
        iters += 1
        if check_stop(x_new, x, eps): return x_new, iters
        x = x_new
    return x, iters


def newton_method(x0, eps=1e-10):
    iters, x = 0, x0
    while iters < 1000:
        x_new = x - F(x) / dF(x)
        iters += 1
        if check_stop(x_new, x, eps): return x_new, iters
        x = x_new
    return x, iters


def chebyshev_method(x0, eps=1e-10):
    iters, x = 0, x0
    while iters < 1000:
        fx, dfx, d2fx = F(x), dF(x), d2F(x)
        x_new = x - fx / dfx - 0.5 * (fx ** 2 * d2fx) / (dfx ** 3)
        iters += 1
        if check_stop(x_new, x, eps): return x_new, iters
        x = x_new
    return x, iters


def secant_method(x_prev, x_curr, eps=1e-10):
    iters = 0
    while iters < 1000:
        x_new = x_curr - F(x_curr) * (x_curr - x_prev) / (F(x_curr) - F(x_prev))
        iters += 1
        if check_stop(x_new, x_curr, eps): return x_new, iters
        x_prev, x_curr = x_curr, x_new
    return x_curr, iters


def parabola_method(x0, x1, x2, eps=1e-10):
    iters = 0
    while iters < 1000:
        f01 = (F(x1) - F(x0)) / (x1 - x0)
        f12 = (F(x2) - F(x1)) / (x2 - x1)
        f012 = (f12 - f01) / (x2 - x0)
        w = f12 + (x2 - x1) * f012
        det = cmath.sqrt(w ** 2 - 4 * F(x2) * f012)
        delta = -2 * F(x2) / (w + det if abs(w + det) > abs(w - det) else w - det)
        x_new = x2 + delta.real
        iters += 1
        if check_stop(x_new, x2, eps): return x_new, iters
        x0, x1, x2 = x1, x2, x_new
    return x2, iters


def inverse_interpolation(x0, x1, x2, eps=1e-10):
    iters = 0
    while iters < 1000:
        y0, y1, y2 = F(x0), F(x1), F(x2)
        x_new = (y1 * y2 * x0) / ((y0 - y1) * (y0 - y2)) + (y0 * y2 * x1) / ((y1 - y0) * (y1 - y2)) + (y0 * y1 * x2) / (
                    (y2 - y0) * (y2 - y1))
        iters += 1
        if check_stop(x_new, x2, eps): return x_new, iters
        x0, x1, x2 = x1, x2, x_new
    return x2, iters


# =============================================================================
# АЛГЕБРАЇЧНА ЧАСТИНА (Пункти 5-9)
# =============================================================================

def point_5_6_setup_and_plot():
    """Підбір рівняння, графік та запис коефіцієнтів (Пункти 5-6)"""
    coeffs = [1.0, -4.0, 6.0, -4.0]  # x^3 - 4x^2 + 6x - 4 = 0

    # Побудова графіка
    x = np.linspace(0.5, 3.5, 100)
    y = x ** 3 - 4 * x ** 2 + 6 * x - 4
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label='F(x) = x³ - 4x² + 6x - 4')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Графік алгебраїчного рівняння (Пункт 5)")
    plt.grid(True)
    plt.legend()
    plt.savefig("polynomial_graph.png")
    print("\n--- Пункт 5: Графік збережено в 'polynomial_graph.png' ---")
    plt.show()

    # Запис у файл
    with open("poly_coeffs.txt", "w") as f:
        f.write(" ".join(map(str, coeffs)))


def read_coeffs(filename):
    """Зчитування коефіцієнтів (Пункт 7)"""
    with open(filename, "r") as f:
        return [float(c) for c in f.read().split()]


def horner_newton(coeffs, x0, eps=1e-10):
    """Метод Ньютона зі схемою Горнера (Пункт 8)"""
    m = len(coeffs) - 1
    x, iters = x0, 0
    while iters < 1000:
        b = [0] * (m + 1)
        b[0] = coeffs[0]
        for i in range(1, m + 1): b[i] = coeffs[i] + x * b[i - 1]

        c = [0] * m
        c[0] = b[0]
        for i in range(1, m): c[i] = b[i] + x * c[i - 1]

        x_new = x - b[m] / c[m - 1]
        iters += 1
        if abs(x_new - x) < eps: return x_new, iters
        x = x_new
    return x, iters


def lin_method(coeffs, alpha0, beta0, eps=1e-10):
    """Метод Ліна для комплексних коренів (Пункт 9)"""
    iters = 0
    while iters < 1000:
        p0, q0 = -2 * alpha0, alpha0 ** 2 + beta0 ** 2
        b2, b1 = coeffs[0], coeffs[1] - p0 * coeffs[0]
        q1 = coeffs[3] / b1
        p1 = (coeffs[2] - q1 * b2) / b1
        alpha1 = -p1 / 2
        beta1 = math.sqrt(max(0, q1 - alpha1 ** 2))
        iters += 1
        if abs(alpha1 - alpha0) < eps and abs(beta1 - beta0) < eps:
            return complex(alpha1, beta1), complex(alpha1, -beta1), iters
        alpha0, beta0 = alpha1, beta1
    return None


# =============================================================================
# ВИКОНАННЯ ВСІХ ПУНКТІВ
# =============================================================================
if __name__ == "__main__":
    # Частина 1 (Трансцендентна)
    approx_roots = point_1_tabulation(-2, 2, 0.1)

    print("\n--- Пункт 4: Результати для знайдених коренів (eps=1e-10) ---")
    for r in approx_roots:
        print(f"\nКорегування кореня біля x ≈ {r:.2f}:")
        print(f"Метод простої ітерації: {simple_iteration(r)}")
        print(f"Метод Ньютона:          {newton_method(r)}")
        print(f"Метод Чебишева:         {chebyshev_method(r)}")
        print(f"Метод хорд:             {secant_method(r - 0.1, r)}")
        print(f"Метод парабол:          {parabola_method(r - 0.1, r, r + 0.1)}")
        print(f"Зворотна інтерполяція:  {inverse_interpolation(r - 0.1, r, r + 0.1)}")

    # Частина 2 (Алгебраїчна)
    point_5_6_setup_and_plot()  # ТУТ ТЕПЕР БУДЕ ГРАФІК (Пункт 5)
    c = read_coeffs("poly_coeffs.txt")

    res_h, it_h = horner_newton(c, 3.0)
    print(f"\n--- Пункт 8: Дійсний корінь (Горнер-Ньютон) ---")
    print(f"x = {res_h:.10f}, ітерацій: {it_h}")

    res_l = lin_method(c, 0.5, 0.5)
    print(f"\n--- Пункт 9: Комплексні корені (Лін) ---")
    print(f"x1,2 = {res_l[0]:.4f}, {res_l[1]:.4f}, ітерацій: {res_l[2]}")