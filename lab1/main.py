import numpy as np
import matplotlib.pyplot as plt


def solve_spline(x, y):
    n = len(x)
    h = np.diff(x)

    A = np.zeros((n, n))
    B = np.zeros(n)

    A[0, 0] = 1
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        B[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    A[n - 1, n - 1] = 1

    c = np.linalg.solve(A, B)

    a = y[:-1]
    d = np.diff(c) / (3 * h)
    b = (np.diff(y) / h) - (h / 3) * (c[1:] + 2 * c[:-1])

    return a, b, c[:-1], d


def evaluate(x_knots, x_val, a, b, c, d):
    idx = np.searchsorted(x_knots, x_val) - 1
    idx = np.clip(idx, 0, len(a) - 1)
    dx = x_val - x_knots[idx]
    return a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3


# Вихідні дані
raw_coords = [(48.164214, 24.536044), (48.164983, 24.534836), (48.165605, 24.534068), (48.166228, 24.532915),
              (48.166777, 24.531927), (48.167326, 24.530884), (48.167011, 24.530061), (48.166053, 24.528039),
              (48.166655, 24.526064), (48.166497, 24.523574), (48.166128, 24.520214), (48.165416, 24.517170),
              (48.164546, 24.514640), (48.163412, 24.512980), (48.162331, 24.511715), (48.162015, 24.509462),
              (48.162147, 24.506932), (48.161751, 24.504244), (48.161197, 24.501793), (48.160580, 24.500537),
              (48.160250, 24.500106)]
elevations = [1250, 1280, 1310, 1350, 1400, 1450, 1480, 1550, 1620, 1700, 1780, 1850, 1920, 1980, 2010, 2030, 2045,
              2055, 2058, 2060, 2061]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    dp = np.radians(lat2 - lat1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


dist = [0]
for i in range(1, len(raw_coords)):
    dist.append(dist[-1] + haversine(*raw_coords[i - 1], *raw_coords[i]))

dist = np.array(dist)
elev = np.array(elevations)

# Візуалізація
plt.figure(figsize=(10, 5))
for n in [10, 15, 20]:
    indices = np.linspace(0, len(dist) - 1, n, dtype=int)
    xn, yn = dist[indices], elev[indices]
    a, b, c, d = solve_spline(xn, yn)
    xp = np.linspace(dist[0], dist[-1], 100)
    yp = [evaluate(xn, xi, a, b, c, d) for xi in xp]
    plt.plot(xp, yp, label=f'Вузлів: {n}')

plt.scatter(dist, elev, color='red', s=10)
plt.legend()
plt.grid(True)
plt.show()