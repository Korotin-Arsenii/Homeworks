import numpy as np
import matplotlib.pyplot as plt


# 1. ГЕНЕРАЦІЯ ДАНИХ (Пункт 1 ходу роботи)
def generate_data(n=100, x_val=2.5):
    # Матриця з діагональним переважанням для збіжності 
    A = np.random.rand(n, n)
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1

    x_true = np.full(n, x_val)
    # Обчислення вектора B 
    b = A @ x_true

    np.savetxt('matrix_A.txt', A)
    np.savetxt('vector_B.txt', b)
    return x_true


# 2. ДОПОМІЖНІ ФУНКЦІЇ (Пункт 2 ходу роботи)
def load_data():
    return np.loadtxt('matrix_A.txt'), np.loadtxt('vector_B.txt')


def vector_norm(v):
    return np.max(np.abs(v))  # Норма L-нескінченність


def matrix_norm(M):
    return np.max(np.sum(np.abs(M), axis=1))  # Рядкова норма


# 3. ІТЕРАЦІЙНІ МЕТОДИ
# Метод простої ітерації
def simple_iteration(A, b, eps=1e-14):
    n = len(b)
    x = np.ones(n)  # Початкове наближення
    tau = 1.0 / matrix_norm(A)  # Параметр збіжності
    history = []

    for k in range(1000):
        x_new = x - tau * (A @ x - b)
        diff = vector_norm(x_new - x)
        history.append(diff)
        if diff < eps:
            break
        x = x_new
    return x, len(history), history


# Метод Якобі
def jacobi_method(A, b, eps=1e-14):
    n = len(b)
    x = np.ones(n)
    D = np.diag(A)
    R = A - np.diag(D)
    history = []

    for k in range(1000):
        x_new = (b - R @ x) / D
        diff = vector_norm(x_new - x)
        history.append(diff)
        if diff < eps:
            break
        x = x_new
    return x, len(history), history


# Метод Зейделя
def gauss_seidel(A, b, eps=1e-14):
    n = len(b)
    x = np.ones(n)
    history = []

    for k in range(1000):
        x_old = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        diff = vector_norm(x - x_old)
        history.append(diff)
        if diff < eps:
            break
    return x, len(history), history


# ГОЛОВНИЙ БЛОК
def main():
    x_true = generate_data()
    A, B = load_data()
    eps0 = 1e-14  # Задана точність

    # Виконання методів
    x_si, it_si, hist_si = simple_iteration(A, B, eps0)
    x_ja, it_ja, hist_ja = jacobi_method(A, B, eps0)
    x_gs, it_gs, hist_gs = gauss_seidel(A, B, eps0)

    print(f"Проста ітерація: {it_si} ітерацій")
    print(f"Метод Якобі: {it_ja} ітерацій")
    print(f"Метод Зейделя: {it_gs} ітерацій")

    # Побудова графіка збіжності
    plt.figure(figsize=(10, 6))
    plt.semilogy(hist_si, label='Проста ітерація')
    plt.semilogy(hist_ja, label='Якобі')
    plt.semilogy(hist_gs, label='Зейдель')
    plt.axhline(y=eps0, color='r', linestyle='--', label='Точність 10^-14')
    plt.title('Збіжність ітераційних методів (Норма нев\'язки)')
    plt.xlabel('Ітерації')
    plt.ylabel('||X(k+1) - X(k)||')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()