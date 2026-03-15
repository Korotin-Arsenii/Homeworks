import requests
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
response = requests.get(url)
data = response.json()
results = data["results"]
n = len(results)

print("Кількість вузлів:", n)
print(" | Latitude | Longitude | Elevation (m)")
for i, p in enumerate(results):
    print(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results])
distances = [0]

for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

distances = np.array(distances)
print(" | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

h = np.diff(distances)
A = np.zeros(n)
C_mat = np.zeros(n)
B = np.zeros(n)
F = np.zeros(n)

for i in range(1, n - 1):
    A[i] = h[i - 1]
    B[i] = h[i]
    C_mat[i] = 2 * (h[i - 1] + h[i])
    F[i] = 3 * ((elevations[i + 1] - elevations[i]) / h[i] - (elevations[i] - elevations[i - 1]) / h[i - 1])

alpha = np.zeros(n)
beta = np.zeros(n)

for i in range(1, n - 1):
    alpha[i] = -B[i] / (A[i] * alpha[i - 1] + C_mat[i])
    beta[i] = (F[i] - A[i] * beta[i - 1]) / (A[i] * alpha[i - 1] + C_mat[i])

c = np.zeros(n)
for i in range(n - 2, 0, -1):
    c[i] = alpha[i] * c[i + 1] + beta[i]

a_coef = elevations[:-1]
b = np.zeros(n - 1)
d = np.zeros(n - 1)

for i in range(n - 1):
    d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    b[i] = (elevations[i + 1] - elevations[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3

print("a_i:", a_coef)
print("b_i:", b)
print("c_i:", c[:-1])
print("d_i:", d)


def eval_spline(x_eval, x_nodes, a, b, c, d):
    y_eval = np.zeros_like(x_eval)
    for idx, x_val in enumerate(x_eval):
        i = np.searchsorted(x_nodes, x_val) - 1
        i = np.clip(i, 0, len(a) - 1)
        dx = x_val - x_nodes[i]
        y_eval[idx] = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
    return y_eval


def build_subset_spline(nodes_count):
    idx = np.linspace(0, n - 1, nodes_count).astype(int)
    x_sub = distances[idx]
    y_sub = elevations[idx]
    n_sub = len(x_sub)
    h_sub = np.diff(x_sub)

    A_s = np.zeros(n_sub)
    C_s = np.zeros(n_sub)
    B_s = np.zeros(n_sub)
    F_s = np.zeros(n_sub)

    for i in range(1, n_sub - 1):
        A_s[i] = h_sub[i - 1]
        B_s[i] = h_sub[i]
        C_s[i] = 2 * (h_sub[i - 1] + h_sub[i])
        F_s[i] = 3 * ((y_sub[i + 1] - y_sub[i]) / h_sub[i] - (y_sub[i] - y_sub[i - 1]) / h_sub[i - 1])

    alpha_s = np.zeros(n_sub)
    beta_s = np.zeros(n_sub)

    for i in range(1, n_sub - 1):
        alpha_s[i] = -B_s[i] / (A_s[i] * alpha_s[i - 1] + C_s[i])
        beta_s[i] = (F_s[i] - A_s[i] * beta_s[i - 1]) / (A_s[i] * alpha_s[i - 1] + C_s[i])

    c_s = np.zeros(n_sub)
    for i in range(n_sub - 2, 0, -1):
        c_s[i] = alpha_s[i] * c_s[i + 1] + beta_s[i]

    a_s = y_sub[:-1]
    b_s = np.zeros(n_sub - 1)
    d_s = np.zeros(n_sub - 1)

    for i in range(n_sub - 1):
        d_s[i] = (c_s[i + 1] - c_s[i]) / (3 * h_sub[i])
        b_s[i] = (y_sub[i + 1] - y_sub[i]) / h_sub[i] - h_sub[i] * (c_s[i + 1] + 2 * c_s[i]) / 3

    return x_sub, a_s, b_s, c_s[:-1], d_s


x_dense = np.linspace(distances[0], distances[-1], 500)
plt.figure(figsize=(10, 5))
for cnt in [10, 15, 20]:
    x_sub, a_s, b_s, c_s, d_s = build_subset_spline(cnt)
    y_spline = eval_spline(x_dense, x_sub, a_s, b_s, c_s, d_s)
    plt.plot(x_dense, y_spline, label=f'{cnt} вузлів')
plt.plot(distances, elevations, 'ko', label='Дані')
plt.legend()
plt.show()

y_approx = eval_spline(x_dense, distances, a_coef, b, c[:-1], d)
y_real = np.interp(x_dense, distances, elevations)
error = np.abs(y_real - y_approx)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x_dense, y_real, 'k', label='f(x)')
ax1.plot(x_dense, y_approx, 'r--', label='Наближення')
ax1.legend()
ax2.plot(x_dense, error, 'b', label='Похибка')
ax2.legend()
plt.show()

total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))
grad = np.gradient(y_approx, x_dense) * 100
energy = 80 * 9.81 * total_ascent

print("Загальна довжина (м):", distances[-1])
print("Сумарний набір (м):", total_ascent)
print("Сумарний спуск (м):", total_descent)
print("Макс підйом (%):", np.max(grad))
print("Макс спуск (%):", np.min(grad))
print("Середній градієнт (%):", np.mean(np.abs(grad)))
print("Робота (Дж):", energy)
print("Робота (кДж):", energy / 1000)
print("Енергія (ккал):", energy / 4184)