import sys

import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt

# dict of gyromagnetic ratios in rad/s/Gs
gammas = {"H": 26753, "N": -2712}


def scalar_product(a, b):
    if len(a) != len(b):
        raise Exception("Operators have different dimensions")
    return sum(a[i] @ b[i] for i in range(len(a)))


def make_spin_operators(n):
    """
    function to make spin operators in a spin system
    which contains n 1/2 spins
    :param n: number of spins
    :return: list of lists of spin operators using sparse matrices
    """
    single_spin = [np.array([[0, 0.5], [0.5, 0]], dtype=np.complex_),
                   np.array([[0, -0.5*1j], [0.5*1j, 0]], dtype=np.complex_),
                   np.array([[0.5, 0], [0, -0.5]], dtype=np.complex_)]
    unity = np.eye(2, dtype=np.complex_)
    I = [single_spin.copy()]

    for i in range(n - 1):
        for j in range(len(I)):
            for k in range(3):
                I[j][k] = np.kron(I[j][k], unity)
        lst = []
        for k in range(3):
            lst.append(np.kron(np.eye(2 ** (i + 1)), single_spin[k]))
        I.append(lst)
    return I


s_h2 = make_spin_operators(2)
p_h2 = 1 / 4 * np.eye(4) - s_h2[0][2] @ s_h2[1][2]
# p_h2 = 1 / 4 * np.eye(4) - scalar_product(s_h2[0], s_h2[1])


def make_relaxation_superoperator(S, n, T):
    unity = np.eye(2 ** n, dtype=np.complex_)
    S_super = []
    for i in range(n):
        S_n = []
        for j in range(3):
            S_n.append(np.kron(S[i][j], unity) - np.kron(unity, S[i][j].T))
        S_super.append(S_n)
    R = np.zeros((4 ** n, 4 ** n), dtype=np.complex_)
    for i in range(n):
        R = R - 1 / (2 * T[i]) * sum(S_super[i][j].T.conjugate() @ S_super[i][j] for j in range(3))
    return R


def make_partial_trace_superoperator(n_complex, n_sub):
    partial_trace_super = np.zeros((4 ** n_sub, 4 ** n_complex), dtype=np.complex_)
    for i in range(2 ** n_sub):
        for j in range(2 ** n_sub):
            for m in range(2 ** (n_complex - n_sub)):
                for k in range(2 ** (n_complex - n_sub)):
                    u = (i * 2 ** (n_complex - n_sub) + m) * 2 ** n_complex + j * 2 ** (n_complex - n_sub) + k
                    v = i * 2 ** n_sub + j
                    if m == k:
                        partial_trace_super[v, u] = 1
    return partial_trace_super


def make_kron_superoperator_ph2(n_complex, n_sub):
    # p_h2 = 1 / 4 * np.eye(4) - scalar_product(s_h2[0], s_h2[1])
    kron_product_super = np.zeros((4 ** n_complex, 4 ** n_sub), dtype=np.complex_)
    for i in range(2 ** n_sub):
        for j in range(2 ** n_sub):
            for m in range(2 ** (n_complex - n_sub)):
                for k in range(2 ** (n_complex - n_sub)):
                    u = (i * 2 ** (n_complex - n_sub) + m) * 2 ** n_complex + j * 2 ** (n_complex - n_sub) + k
                    v = i * 2 ** n_sub + j
                    kron_product_super[u, v] = p_h2[m, k]
    return kron_product_super


def make_hamiltonian_superoperator_complex(S, d, J, b, n_N, n_H, w_1_N, w_1_H, w_rf_N, w_rf_H):
    # N spins first and H spins second
    delta_w = np.zeros(3)
    for i in range(3):
        if i < 1:
            delta_w[i] = -gammas["N"] * b * (1 + d[i] * 10 ** -6) - w_rf_N
        else:
            delta_w[i] = gammas["H"] * b * (1 + d[i] * 10 ** -6) - w_rf_H
    H_z = delta_w[0] * S[0][2] - delta_w[1] * S[1][2] - delta_w[2] * S[2][2] + \
          w_1_N * S[0][0] - w_1_H * (S[2][0] + S[1][0])
    H_j_hh = 2 * np.pi * J[1][2] * (S[1][0] @ S[2][0] + S[1][1] @ S[2][1] + S[1][2] @ S[2][2])
    H_j_nh = 2 * np.pi * J[0][1] * S[0][2] @ S[1][2]
    H = H_z + H_j_nh + H_j_hh
    return np.kron(H, np.eye(2 ** 3)) - np.kron(np.eye(2 ** 3), H.T)


def make_hamiltonian_superoperator_free(S, d, J, b, n_N, n_H, w_1_N, w_1_H, w_rf_N, w_rf_H):
    # N spins first and H spins second
    delta_w = -gammas["N"] * b * (1 + d[0] * 10 ** -6) - w_rf_N
    H = delta_w * S[0][2] + w_1_N * S[0][0]
    return np.kron(H, np.eye(2 ** 1)) - np.kron(np.eye(2 ** 1), H.T)


def make_solution_matrix(H_c, H_f, R_c, R_f, n_c, n_f, k_d, S_trace, S_kron, c_s_ratio):
    upper_block = np.hstack([-1j * H_f + R_f - k_d * c_s_ratio * np.eye(4 ** n_f), k_d * S_trace])
    bottom_block = np.hstack([k_d * c_s_ratio * S_kron, -1j * H_c + R_c - k_d * np.eye(4 ** n_c)])
    full_matrix = np.vstack([upper_block, bottom_block])
    return full_matrix


number_of_N_spins = 1
number_of_H_spins_free = 0
number_of_H_spins_complex = 2
full_number_of_spins_complex = number_of_N_spins + number_of_H_spins_complex
full_number_of_spins_free = number_of_N_spins + number_of_H_spins_free

I_complex = make_spin_operators(full_number_of_spins_complex)
I_free = make_spin_operators(full_number_of_spins_free)

T_complex = np.array([3, 1, 1])
T_free = np.array([30])

chemical_shift_complex = [199, -29.1, -27.2]
# chemical_shift_complex = [198.5, -22.31929, -23]
chemical_shift_free = [250]

J_complex = np.zeros((3, 3))
J_complex[0][1] = -20
J_complex[1][2] = -7
J_free = np.zeros((3, 3))


magnetic_field = 9.4 * 10 ** 4

w_res_H_1 = gammas["H"] * magnetic_field * (1 + (chemical_shift_complex[1]) * 10 ** -6)
w_res_N = -gammas["N"] * magnetic_field * (1 + chemical_shift_complex[0] * 10 ** -6)
w_rf_H = gammas["H"] * magnetic_field * (1 + (chemical_shift_complex[1] + 0.2) * 10 ** -6)
w_1_H = 2 * np.pi * 20
w_1_N = 2 * np.pi * 20

'''
w_offset_H_1 = gammas["H"] * magnetic_field * (1 + chemical_shift_complex[1] * 10 ** -6) - w_rf_H
w_offset_H_2 = gammas["H"] * magnetic_field * (1 + chemical_shift_complex[2] * 10 ** -6) - w_rf_H
nu_eff_H_1 = np.sqrt(w_offset_H_1 ** 2 + w_1_H ** 2) / 2 / np.pi
nu_eff_H_2 = np.sqrt(w_offset_H_2 ** 2 + w_1_H ** 2) / 2 / np.pi

print('nu_eff_H_1 = ' + str(nu_eff_H_1) + ' Hz')
print('nu_eff_H_2 = ' + str(nu_eff_H_2) + ' Hz')
'''

k_d = 5
catalyst_to_substrate_ratio = 3 / 30
complex_norm_factor = catalyst_to_substrate_ratio / (1 + catalyst_to_substrate_ratio)
free_norm_factor = 1 / (1 + catalyst_to_substrate_ratio)

unity_complex = np.kron(np.eye(2 ** number_of_N_spins) / 2 ** number_of_N_spins, p_h2)
unity_free = np.eye(2 ** full_number_of_spins_free) / 2 ** full_number_of_spins_free
p_free_initial_vector = np.reshape(free_norm_factor * unity_free, (4 ** full_number_of_spins_free, 1))
p_complex_initial_vector = np.reshape(complex_norm_factor * unity_complex, (4 ** full_number_of_spins_complex, 1))
p_full_initial_vector = np.vstack([p_free_initial_vector, p_complex_initial_vector])

R_complex = make_relaxation_superoperator(I_complex, full_number_of_spins_complex, T_complex)
R_free = make_relaxation_superoperator(I_free, full_number_of_spins_free, T_free)
S_trace = make_partial_trace_superoperator(full_number_of_spins_complex, full_number_of_spins_free)
S_kron = make_kron_superoperator_ph2(full_number_of_spins_complex, full_number_of_spins_free)

'''
# CYCLES
t_exchange = 1
t_ev = 0.1
N = 1000
w_offset_N = np.linspace(-2 * np.pi * 3500, 2 * np.pi * 3600, N)
m_z = np.zeros(N)
for i in range(N):
    w_rf_N = w_res_N + w_offset_N[i]
    H_super_complex = make_hamiltonian_superoperator(I_complex, chemical_shift_complex, J_complex,
                                                     magnetic_field, number_of_N_spins, number_of_H_spins_complex,
                                                     w_1_N, w_1_H, w_rf_N, w_rf_H)
    H_super_free = make_hamiltonian_superoperator(I_free, chemical_shift_free, J_free,
                                                     magnetic_field, number_of_N_spins, number_of_H_spins_free,
                                                     w_1_N, w_1_H, w_rf_N, w_rf_H)
    A = make_solution_matrix(H_super_complex, H_super_free, R_complex, R_free, full_number_of_spins_complex,
                             full_number_of_spins_free, k_d, S_trace, S_kron, catalyst_to_substrate_ratio)
    H_super_complex_exchange = make_hamiltonian_superoperator(I_complex, chemical_shift_complex, J_complex,
                                                     magnetic_field, number_of_N_spins, number_of_H_spins_complex,
                                                     0, 0, w_rf_N, w_rf_H)
    H_super_free_exchange = make_hamiltonian_superoperator(I_free, chemical_shift_free, J_free,
                                                  magnetic_field, number_of_N_spins, number_of_H_spins_free,
                                                  0, 0, w_rf_N, w_rf_H)
    B = make_solution_matrix(H_super_complex_exchange, H_super_free_exchange, R_complex, R_free,
                             full_number_of_spins_complex,full_number_of_spins_free, k_d, S_trace,
                             S_kron, catalyst_to_substrate_ratio)
    p_full_current_vector = np.linalg.matrix_power(la.expm(A * t_ev) @ la.expm(B * t_exchange), 30) @ \
                            p_full_initial_vector
    p_free_current_matrix = np.reshape(p_full_current_vector[:4 ** full_number_of_spins_free],
                                       (2 ** full_number_of_spins_free, 2 ** full_number_of_spins_free))
    m_z[i] = np.real(np.trace((I_free[0][2]) @ p_free_current_matrix) / np.trace(p_free_current_matrix))
plt.plot(w_offset_N / 2 / np.pi / (2712 / 26753 * 400) + 0, m_z)
plt.show()
'''
'''
# NO CYCLES
t_b = 10
N = 1000
w_offset_N = np.linspace(-2 * np.pi * 2500, 2 * np.pi * 2500, N)
m_z_free = np.zeros(N)
m_z_bound = np.zeros(N)

for i in range(N):
    w_rf_N = w_res_N + w_offset_N[i]
    H_super_complex = make_hamiltonian_superoperator(I_complex, chemical_shift_complex, J_complex,
                                                     magnetic_field, number_of_N_spins, number_of_H_spins_complex,
                                                     w_1_N, w_1_H, w_rf_N, w_rf_H)
    H_super_free = make_hamiltonian_superoperator(I_free, chemical_shift_free, J_free,
                                                     magnetic_field, number_of_N_spins, number_of_H_spins_free,
                                                     w_1_N, w_1_H, w_rf_N, w_rf_H)
    A = make_solution_matrix(H_super_complex, H_super_free, R_complex, R_free, full_number_of_spins_complex,
                             full_number_of_spins_free, k_d, S_trace, S_kron, catalyst_to_substrate_ratio)
    p_full_current_vector = la.expm(A * t_b)  @ p_full_initial_vector
    p_bound_current_matrix = np.reshape(p_full_current_vector[4 ** full_number_of_spins_free:
                                                              4 ** full_number_of_spins_free + 4 ** full_number_of_spins_complex],
                                        (2 ** full_number_of_spins_complex, 2 ** full_number_of_spins_complex))
    p_free_current_matrix = np.reshape(p_full_current_vector[:4 ** full_number_of_spins_free],
                                       (2 ** full_number_of_spins_free, 2 ** full_number_of_spins_free))
    m_z_free[i] = np.real(np.trace(I_free[0][2] @ p_free_current_matrix) / np.trace(p_free_current_matrix))
    m_z_bound[i] = np.real(np.trace(I_complex[0][2] @ p_bound_current_matrix) / np.trace(p_bound_current_matrix))
plt.plot(w_offset_N / 2 / np.pi, m_z_free)
plt.plot(w_offset_N / 2 / np.pi, m_z_bound)
plt.show()
'''
t_b = 100
N = 1000
w_offset_N = np.linspace(-2 * np.pi * 200, 2 * np.pi * 200, N)
m_z_free = np.zeros(N)
with open("!!test.txt", 'w') as f:
    sys.stdout = f
    for j in range(N):
        w_rf_N = w_res_N + w_offset_N[j]
        H_super_complex = make_hamiltonian_superoperator_complex(I_complex, chemical_shift_complex, J_complex,
                                                         magnetic_field, number_of_N_spins, number_of_H_spins_complex,
                                                         w_1_N, w_1_H, w_rf_N, w_rf_H)
        H_super_free = make_hamiltonian_superoperator_free(I_free, chemical_shift_free, J_free,
                                                         magnetic_field, number_of_N_spins, number_of_H_spins_free,
                                                         w_1_N, w_1_H, w_rf_N, w_rf_H)
        A = make_solution_matrix(H_super_complex, H_super_free, R_complex, R_free, full_number_of_spins_complex,
                                 full_number_of_spins_free, k_d, S_trace, S_kron, catalyst_to_substrate_ratio)
        p_full_current_vector = la.expm(A * t_b)  @ p_full_initial_vector
        p_free_current_matrix = np.reshape(p_full_current_vector[:4 ** full_number_of_spins_free],
                                           (2 ** full_number_of_spins_free, 2 ** full_number_of_spins_free))
        m_z_free[j] = np.real(np.trace(I_free[0][2] @ p_free_current_matrix) / np.trace(p_free_current_matrix))
        print(w_offset_N[j] / 2 / np.pi, m_z_free[j])

plt.plot(w_offset_N / 2 / np.pi, m_z_free)
plt.show()
'''
t_b = 10
N = 1000
N_h = 100
w_offset_N = np.linspace(-2 * np.pi * 200, 2 * np.pi * 200, N)
w_offset_H = np.linspace(-2 * np.pi * 200, 2 * np.pi * 200, N_h)
m_z_free = np.zeros((N_h, N))
with open("TRANS-CIS_6hz.txt", 'w') as f:
    sys.stdout = f
    for i in range(N_h):
        w_rf_H = w_res_H_1 + w_offset_H[i]
        for j in range(N):
            w_rf_N = w_res_N + w_offset_N[j]
            H_super_complex = make_hamiltonian_superoperator_complex(I_complex, chemical_shift_complex, J_complex,
                                                             magnetic_field, number_of_N_spins, number_of_H_spins_complex,
                                                             w_1_N, w_1_H, w_rf_N, w_rf_H)
            H_super_free = make_hamiltonian_superoperator_free(I_free, chemical_shift_free, J_free,
                                                             magnetic_field, number_of_N_spins, number_of_H_spins_free,
                                                             w_1_N, w_1_H, w_rf_N, w_rf_H)
            A = make_solution_matrix(H_super_complex, H_super_free, R_complex, R_free, full_number_of_spins_complex,
                                     full_number_of_spins_free, k_d, S_trace, S_kron, catalyst_to_substrate_ratio)
            p_full_current_vector = la.expm(A * t_b)  @ p_full_initial_vector
            p_free_current_matrix = np.reshape(p_full_current_vector[:4 ** full_number_of_spins_free],
                                               (2 ** full_number_of_spins_free, 2 ** full_number_of_spins_free))
            m_z_free[i][j] = np.real(np.trace(I_free[0][2] @ p_free_current_matrix) / np.trace(p_free_current_matrix))
            print(w_offset_H[i] / 2 / np.pi, w_offset_N[j] / 2 / np.pi, m_z_free[i][j])
'''
'''
N = 1000
w_offset_N_sweep = np.linspace(-2 * np.pi * 3500, -2 * np.pi * 3600, N)
#w_offset_N_sweep = np.linspace(-2 * np.pi * 3560, -2 * np.pi * 3560, N)
N_tau = 100
m_z_free = np.zeros(N_tau)
m_z_bound = np.zeros(N_tau)
tau = np.linspace(0, 1, N_tau)
p_full_current_vector = p_full_initial_vector
for i in range(N_tau):
    dt = tau[i] / N
    for j in range(N):
        w_rf_N = w_res_N + w_offset_N_sweep[j]
        H_super_complex = make_hamiltonian_superoperator(I_complex, chemical_shift_complex, J_complex,
                                                         magnetic_field, number_of_N_spins, number_of_H_spins_complex,
                                                         w_1_N, w_1_H, w_rf_N, w_rf_H)
        H_super_free = make_hamiltonian_superoperator(I_free, chemical_shift_free, J_free,
                                                         magnetic_field, number_of_N_spins, number_of_H_spins_free,
                                                         w_1_N, w_1_H, w_rf_N, w_rf_H)
        A = make_solution_matrix(H_super_complex, H_super_free, R_complex, R_free, full_number_of_spins_complex,
                                 full_number_of_spins_free, k_d, S_trace, S_kron, catalyst_to_substrate_ratio)
        p_full_current_vector = la.expm(A * dt)  @ p_full_current_vector
    p_bound_current_matrix = np.reshape(p_full_current_vector[4 ** full_number_of_spins_free:
                                        4 ** full_number_of_spins_free + 4 ** full_number_of_spins_complex],
                                       (2 ** full_number_of_spins_complex, 2 ** full_number_of_spins_complex))
    p_free_current_matrix = np.reshape(p_full_current_vector[:4 ** full_number_of_spins_free],
                                       (2 ** full_number_of_spins_free, 2 ** full_number_of_spins_free))
    m_z_free[i] = np.real(np.trace(I_free[0][2] @ p_free_current_matrix) / np.trace(p_free_current_matrix))
    m_z_bound[i] = np.real(np.trace(I_complex[0][2] @ p_bound_current_matrix) / np.trace(p_bound_current_matrix))
    p_full_current_vector = p_full_initial_vector

plt.plot(tau, m_z_free)
plt.plot(tau, m_z_bound)
plt.show()
'''