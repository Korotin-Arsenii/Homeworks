import numpy as np
import matplotlib.pyplot as plt


# 1. ГЕНЕРАЦІЯ ТА ЗБЕРЕЖЕННЯ ДАНИХ
def step1_generate_data(n=100, x_val=2.5):
    A = np.random.rand(n, n)
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1

    x_true = np.full(n, x_val)
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b[i] += A[i, j] * x_true[j]

    np.savetxt('matrix_A.txt', A)
    np.savetxt('vector_B.txt', b)
    # Повертаємо x_true для розрахунку похибки
    return n, x_true


# 2. ФУНКЦІЇ ДЛЯ РОБОТИ З LU-РОЗКЛАДОМ
def load_data():
    A = np.loadtxt('matrix_A.txt')
    B = np.loadtxt('vector_B.txt')
    return A, B


def vector_norm(v):
    return np.max(np.abs(v))


def mat_vec_mult(A, x):
    n = len(x)
    res = np.zeros(n)
    for i in range(n):
        for j in range(n):
            res[i] += A[i, j] * x[j]
    return res


def lu_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    return L, U


def solve_lu(L, U, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x


# 3-5. РОЗВ'ЯЗОК, ОЦІНКА ТА УТОЧНЕННЯ
def main():
    n, x_true = step1_generate_data()
    A, B = load_data()

    # Списки для збереження значень для графіка
    residual_history = []
    error_history = []

    # LU-розклад
    L, U = lu_decomposition(A)
    np.savetxt('matrix_L.txt', L)
    np.savetxt('matrix_U.txt', U)

    # Початковий розв'язок через LU
    x_k = solve_lu(L, U, B)

    # Функції оцінки
    def get_residual(curr_x):
        ax = mat_vec_mult(A, curr_x)
        return vector_norm(ax - B)

    def get_error(curr_x):
        return vector_norm(curr_x - x_true)

    # Запис початкових значень (0 ітерація)
    residual_history.append(get_residual(x_k))
    error_history.append(get_error(x_k))

    # 5. Ітераційне уточнення до eps0 = 10^-14
    eps0 = 1e-14
    iterations = 0

    while residual_history[-1] > eps0 and iterations < 20:  # Обмеження для наочності графіка
        r = B - mat_vec_mult(A, x_k)  # Вектор нев'язки
        d = solve_lu(L, U, r)  # Поправка
        x_k += d

        iterations += 1
        residual_history.append(get_residual(x_k))
        error_history.append(get_error(x_k))

    print(f"Ітерацій уточнення: {iterations}")
    print(f"Фінальна точність (нев'язка): {residual_history[-1]:.2e}")
    print(f"Фінальна похибка: {error_history[-1]:.2e}")

    # ПОБУДОВА ГРАФІКА
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(residual_history)), residual_history, 'o-', label="Норма нев'язки $||AX^{(k)} - B||$")
    plt.semilogy(range(len(error_history)), error_history, 's--', label="Норма похибки $||X^{(k)} - X_{true}||$")

    plt.axhline(y=eps0, color='r', linestyle=':', label='Цільова точність $10^{-14}$')

    plt.title('Збіжність ітераційного уточнення розв\'язку')
    plt.xlabel('Номер ітерації')
    plt.ylabel('Величина норми (Log scale)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()