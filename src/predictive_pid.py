import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# Simulation parameters
T = 0.005         # Sampling time [s]
zc = 0.814        # CoM height [m]
g = 9.81          # gravity [m/s^2]
steps = 1000          # total time steps

# Discrete-time state-space matrices (3rd-order Taylor expansion)
A = np.array([
    [1, T, 0.5 * T**2],
    [0, 1, T],
    [0, 0, 1]
])
B = np.array([1/6 * T**3, 0.5 * T**2, T])
C = np.array([1, 0, -zc / g])  # ZMP output

# Predictive PID gains (tuned)
k0, k1, k2 = 3.65496980e-03, 1.18703219e-03, -5.69411295e-03

# Initialize states and control
x_state = np.array([0.0, 0.0, 0.0])
y_state = np.array([0.0, 0.0, 0.0])
u_x, u_y = 0.0, 0.0

# Reference trajectories (sinusoidal)
t = np.linspace(0, steps * T, steps)
x_ref = 0.1 * np.sin(2 * np.pi * t)         # forward motion
y_ref = 0.05 * np.sin(4 * np.pi * t)        # lateral sway

# Error buffers
e_x, e_y = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

# ZMP bounds (support polygon)
zmp_x_bounds = (-0.05, 0.05)  # ±5 cm (toe to heel)
zmp_y_bounds = (-0.03, 0.03)  # ±3 cm (left to right)
zmp_violations = []          # list of violations

# Recording history
x_list, xzmp_list, ux_list = [], [], []
y_list, yzmp_list, uy_list = [], [], []

for k in range(steps):
    # Error calculation
    e_x[0] = x_ref[k] - x_state[0]
    e_y[0] = y_ref[k] - y_state[0]

    print(e_x,e_y)

    # Predictive PID update
    du_x = k0 * e_x[0] + k1 * e_x[1] + k2 * e_x[2]
    du_y = k0 * e_y[0] + k1 * e_y[1] + k2 * e_y[2]
    u_x += du_x
    u_y += du_y

    # State update
    x_state = A @ x_state + B * u_x
    y_state = A @ y_state + B * u_y

    # ZMP calculation
    zmp_x = C @ x_state
    zmp_y = C @ y_state

    # ZMP constraint check
    if not (zmp_x_bounds[0] <= zmp_x <= zmp_x_bounds[1]):
        zmp_violations.append((k * T, 'X', zmp_x))
    if not (zmp_y_bounds[0] <= zmp_y <= zmp_y_bounds[1]):
        zmp_violations.append((k * T, 'Y', zmp_y))

    # Record data
    x_list.append(x_state[0])
    xzmp_list.append(zmp_x)
    ux_list.append(u_x)

    y_list.append(y_state[0])
    yzmp_list.append(zmp_y)
    uy_list.append(u_y)

    # Shift error buffers
    e_x[2], e_x[1] = e_x[1], e_x[0]
    e_y[2], e_y[1] = e_y[1], e_y[0]

# Report violations
if zmp_violations:
    print(f"\nZMP Violations Detected: {len(zmp_violations)}")
    for t_val, axis, val in zmp_violations[:5]:  # first 5
        print(f"At t={t_val:.3f}s, ZMP {axis} = {val:.4f} m (out of bounds)")
else:
    print("\nZMP remained within stable bounds throughout.")

# Plotting
plt.figure(figsize=(14, 8))

# CoM positions
plt.subplot(3, 2, 1)
plt.plot(t, x_list, label='x CoM')
plt.plot(t, x_ref, label='x Ref', linestyle='--')
plt.title('X CoM Trajectory')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(t, y_list, label='y CoM')
plt.plot(t, y_ref, label='y Ref', linestyle='--')
plt.title('Y CoM Trajectory')
plt.legend()

# ZMP outputs with bounds
plt.subplot(3, 2, 3)
plt.plot(t, xzmp_list, label='ZMP X')
plt.axhline(zmp_x_bounds[0], color='r', linestyle='--', label='Bounds')
plt.axhline(zmp_x_bounds[1], color='r', linestyle='--')
plt.title('ZMP in X (with bounds)')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(t, yzmp_list, label='ZMP Y')
plt.axhline(zmp_y_bounds[0], color='r', linestyle='--', label='Bounds')
plt.axhline(zmp_y_bounds[1], color='r', linestyle='--')
plt.title('ZMP in Y (with bounds)')
plt.legend()

# Control inputs
plt.subplot(3, 2, 5)
plt.plot(t, ux_list)
plt.title('Control Input X (Jerk)')

plt.subplot(3, 2, 6)
plt.plot(t, uy_list)
plt.title('Control Input Y (Jerk)')

plt.tight_layout()
plt.show()
