import numpy as np
import matplotlib.pyplot as plt


# 1. Система рівнянь та цільові функції
def f1(x): return x[0] ** 2 + x[1] ** 2 - 4


def f2(x): return x[1] - x[0] ** 2


def phi_sys(x):
    return f1(x) ** 2 + f2(x) ** 2


def rosenbrock(x):
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2


# 2. Метод Хука-Дживса
def hooke_jeeves(func, x0, step, q=2.0, eps1=1e-5, eps2=1e-5, p=2.0):
    x_base = np.array(x0, dtype=float)
    x_curr = np.copy(x_base)
    step = np.array(step, dtype=float)
    n = len(x0)
    trajectory = [np.copy(x_base)]

    def explore(x_start, current_step, reduce_step=True):
        x_new = np.copy(x_start)
        for i in range(n):
            f_best = func(x_new)
            x_tmp = np.copy(x_new)

            x_tmp[i] += current_step[i]
            if func(x_tmp) < f_best:
                x_new = np.copy(x_tmp)
                continue

            x_tmp[i] -= 2 * current_step[i]
            if func(x_tmp) < f_best:
                x_new = np.copy(x_tmp)
                continue

            if reduce_step:
                while current_step[i] >= eps1:
                    current_step[i] /= q
                    x_tmp = np.copy(x_new)
                    x_tmp[i] += current_step[i]
                    if func(x_tmp) < func(x_new):
                        x_new = np.copy(x_tmp)
                        break
                    x_tmp[i] -= 2 * current_step[i]
                    if func(x_tmp) < func(x_new):
                        x_new = np.copy(x_tmp)
                        break
        return x_new, current_step

    steps_count = 0
    while True:
        steps_count += 1
        x_next, step = explore(x_base, step, reduce_step=True)

        if np.all(step < eps1) or abs(func(x_next) - func(x_base)) < eps2:
            break
        if np.array_equal(x_next, x_base):
            break

        # Пошук по зразку
        x_p = x_next + p * (x_next - x_base)
        x_p_explored, _ = explore(x_p, step, reduce_step=False)

        if func(x_p_explored) < func(x_next):
            x_base = np.copy(x_next)
            x_curr = np.copy(x_p_explored)
        else:
            x_base = np.copy(x_next)
            x_curr = np.copy(x_next)

        trajectory.append(np.copy(x_curr))

    return x_curr, trajectory, steps_count


# 3. Тестування на функції Розенброка
x0_ros = [-1.2, 0.0]
step_ros = [0.5, 0.5]
res_ros, traj_ros, count_ros = hooke_jeeves(rosenbrock, x0_ros, step_ros)

# 4. Знаходження розв'язку заданої системи рівнянь
x0_sys = [1.0, 1.0]
step_sys = [0.5, 0.5]
res_sys, traj_sys, count_sys = hooke_jeeves(phi_sys, x0_sys, step_sys)

# 5. Збереження траєкторії у файл та вивід результатів
with open('trajectory.txt', 'w') as f:
    f.write("Траєкторія спуску (Система рівнянь):\n")
    for idx, pt in enumerate(traj_sys):
        f.write(f"Крок {idx}: x1={pt[0]:.5f}, x2={pt[1]:.5f}, Phi={phi_sys(pt):.5f}\n")

# Побудова графіків для системи рівнянь
x_parabola = np.linspace(-3, 3, 400)
x_circle = np.linspace(-2, 2, 400) # Обмеження від -2 до 2 для кола

y1 = np.sqrt(4 - x_circle**2)
y1_neg = -np.sqrt(4 - x_circle**2)
y2 = x_parabola**2

plt.figure(figsize=(6, 6))
plt.plot(x_circle, y1, 'b', label='$x_1^2 + x_2^2 = 4$')
plt.plot(x_circle, y1_neg, 'b')
plt.plot(x_parabola, y2, 'r', label='$x_2 = x_1^2$')

traj_x = [p[0] for p in traj_sys]
traj_y = [p[1] for p in traj_sys]
plt.plot(traj_x, traj_y, 'go-', markersize=4, label='Траєкторія пошуку')
plt.scatter(res_sys[0], res_sys[1], color='k', zorder=5, label='Знайдений розв\'язок')
plt.legend()
plt.grid(True)
plt.title("Графіки рівнянь та траєкторія спуску")
plt.savefig('plot.png')
plt.show()

print("--- Відповідь ---")
print(f"Мінімум функції Розенброка: X* = [{res_ros[0]:.5f}, {res_ros[1]:.5f}], Кроків: {count_ros}")
print(f"Розв'язок системи рівнянь: X* = [{res_sys[0]:.5f}, {res_sys[1]:.5f}], Кроків: {count_sys}")
print(f"Точність (значення цільової функції): {phi_sys(res_sys):.2e}")