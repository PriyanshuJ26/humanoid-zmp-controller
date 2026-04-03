import numpy as np

# Parameters (from simulation section)
T = 0.005  # sampling time [s]
zc = 0.814  # height of center of mass [m]
g = 9.81  # gravity [m/s^2]

# Continuous system matrices (jerk input model)
A_c = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
])
B_c = np.array([[0], [0], [1]])
C_c = np.array([[1, 0, -zc/g]])

# Discretization using Taylor expansion (as in Eq. 6)
A_d = np.array([
    [1, T, T**2 / 2],
    [0, 1, T],
    [0, 0, 1]
])
B_d = np.array([[T**3 / 6], [T**2 / 2], [T]])
C_d = C_c  # No change for output equation

# Form the augmented state-space model
# x_aug = [x; y_integrator] to embed integral action
A_aug = np.block([
    [A_d, np.zeros((3,1))],
    [C_d @ A_d, np.array([[1]])]
])
B_aug = np.vstack((B_d, C_d @ B_d))
C_aug = np.array([[0, 0, 0, 1]])  # Output = integrated error

# Cost matrices (from paper)
Q = 1e-6
R = 1e-6

# Prediction and control horizon
Np = 10000
Nc = 10000

# Construct F and Phi matrices
F = np.zeros((Np, A_aug.shape[1]))
Phi = np.zeros((Np, Nc))

A_pow = np.eye(A_aug.shape[0])
for i in range(Np):
    A_pow = A_pow @ A_aug
    F[i, :] = C_aug @ A_pow
    for j in range(Nc):
        if i - j >= 0:
            Aj = np.linalg.matrix_power(A_aug, i-j)
            Phi[i, j] = C_aug @ Aj @ B_aug

# Compute the optimal gain vector (simplified LQR form)
K_pid = np.linalg.inv(Phi.T @ Phi + R * np.eye(Nc)) @ Phi.T

# Extract k0, k1, k2
k0, k1, k2 = K_pid[0], K_pid[1], K_pid[2]
print(k0.shape)

# Print results
print("Discrete-time A matrix (A_d):\n", A_d)
print("Discrete-time B matrix (B_d):\n", B_d)
print("Output matrix (C_d):\n", C_d)
print("Gain vector [k0, k1, k2]:", k0, k1, k2)
