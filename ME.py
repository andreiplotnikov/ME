import scipy
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
import time

c = 3e10
mass = 9.1e-28
Planck = 6.6e-27
Bolzman = 1.38e-16
el_c = 4.8e-10


def H_function(v, a):
    """
    What is the function?
    :param v:
    :param a:
    :return:
    """
    return np.real(scipy.special.wofz(v + a * 1j))


def L_function(v, a):
    """

    :param v:
    :param a:
    :return:
    """
    return np.imag(scipy.special.wofz(v + a * 1j))


def ME(line_vec, param_vec, x):
    """
    Function description
     wl0 - Angstroms
    mu - cos LOS angle
    B - gauss
    theta - rad
    xi - rad
    D - mAngstroms
    gamma - mAngstroms
    Dop_shift - mAngstroms
    x - mAngstroms


    :param line_vec:
    :param param_vec:
    :param x:
    :return:
    
    Maybe better to change param type to dict not list?
    """

    # constant parameters
    wl0 = line_vec[0] * 1e-8
    g = line_vec[1]
    mu = line_vec[2]

    # parameters for inversion
    B = param_vec[0]
    theta = param_vec[1]  # inclination
    xi = param_vec[2]  # azimuth

    D = param_vec[3] * 1e-11  # Doppler width
    gamma = param_vec[4] * 1e-11  # damping
    etta_0 = param_vec[5]  # line strength

    cont_int = param_vec[6]  # contiuum intensity

    # S_0 = param_vec[6]      #Source function
    betta = param_vec[7]  # Source function decrement

    Dop_shift = param_vec[8] * 1e-11  # Doppler shift

    # units
    v = (x * 1e-11 - Dop_shift) / D
    a = gamma / D
    v_b = B * wl0 * wl0 * el_c / (4 * np.pi * mass * c * c * D)
    S_0 = cont_int / (1 + betta * mu)
    Df = D * c / (wl0 * wl0)

    ka_L = etta_0

    etta_p = H_function(v, a) / np.sqrt(np.pi)
    etta_b = H_function(v - g * v_b, a) / np.sqrt(np.pi)
    etta_r = H_function(v + g * v_b, a) / np.sqrt(np.pi)

    rho_p = L_function(v, a) / np.sqrt(np.pi)
    rho_b = L_function(v - g * v_b, a) / np.sqrt(np.pi)
    rho_r = L_function(v + g * v_b, a) / np.sqrt(np.pi)

    h_I = 0.5 * (etta_p * np.sin(theta) * np.sin(theta) + 0.5 * (etta_b + etta_r) * (1 + np.cos(theta) * np.cos(theta)))
    h_Q = 0.5 * (etta_p - 0.5 * (etta_b + etta_r)) * np.sin(theta) * np.sin(theta) * np.cos(2 * xi)
    h_U = 0.5 * (etta_p - 0.5 * (etta_b + etta_r)) * np.sin(theta) * np.sin(theta) * np.sin(2 * xi)
    h_V = 0.5 * (etta_r - etta_b) * np.cos(theta)
    r_Q = 0.5 * (rho_p - 0.5 * (rho_b + rho_r)) * np.sin(theta) * np.sin(theta) * np.cos(2 * xi)
    r_U = 0.5 * (rho_p - 0.5 * (rho_b + rho_r)) * np.sin(theta) * np.sin(theta) * np.sin(2 * xi)
    r_V = 0.5 * (rho_r - rho_b) * np.cos(theta)

    k_I = ka_L * h_I
    k_Q = ka_L * h_Q
    k_U = ka_L * h_U
    k_V = ka_L * h_V
    f_Q = ka_L * r_Q
    f_U = ka_L * r_U
    f_V = ka_L * r_V

    det = np.power(1 + k_I, 4) + np.power(1 + k_I, 2) * (
                f_Q * f_Q + f_U * f_U + f_V * f_V - k_Q * k_Q - k_U * k_U - k_V * k_V) - np.power(
        k_Q * f_Q + k_U * f_U + k_V * f_V, 2)

    I = 1 - betta * mu / (1 + betta * mu) * (
                1 - (1 + k_I) * ((1 + k_I) * (1 + k_I) + f_Q * f_Q + f_U * f_U + f_V * f_V) / det)
    V = betta * mu / (1 + betta * mu) * ((1 + k_I) * (1 + k_I) * k_V + f_V * (k_Q * f_Q + k_U * f_U + k_V * f_V)) / det
    U = -betta * mu / (1 + betta * mu) / det * (
                (1 + k_I) * (1 + k_I) * k_U - (1 + k_I) * (k_V * f_Q - k_Q * f_V) + f_U * (
                    k_Q * f_Q + k_U * f_U + k_V * f_V))
    Q = -betta * mu / (1 + betta * mu) / det * (
                (1 + k_I) * (1 + k_I) * k_Q - (1 + k_I) * (k_U * f_V - k_V * f_U) + f_Q * (
                    k_Q * f_Q + k_U * f_U + k_V * f_V))

    I *= cont_int
    Q *= cont_int
    U *= cont_int
    V *= cont_int

    return np.transpose(np.array([I, Q, U, V]))


def ME_ff(line_vec, param_vec, x):
    """

    :param line_vec:
    :param param_vec:
    :param x:
    :return:
    """
    # constant parameters
    wl0 = line_vec[0]
    g = line_vec[1]
    mu = line_vec[2]

    # parameters for inversion
    B = param_vec[0]
    theta = param_vec[1]  # inclination
    xi = param_vec[2]  # azimuth

    D = param_vec[3]  # Doppler width
    gamma = param_vec[4]  # damping
    etta_0 = param_vec[5]  # line strength

    cont_int = param_vec[6]  # Source function
    betta = param_vec[7]  # Source function decrement

    Dop_shift = param_vec[8]  # Doppler shift

    filling_factor = param_vec[9]
    stray_shift = param_vec[10]

    return (filling_factor * ME([wl0, g, mu], [B, theta, xi, D, gamma, etta_0, cont_int, betta, Dop_shift], x) + (
                1 - filling_factor) * ME([wl0, g, mu], [0, theta, xi, D, gamma, etta_0, cont_int, betta, stray_shift],
                                         x))


def ME_con(line_vec, param_vec, argument):
    """
    
    :param line_vec:
    :param param_vec:
    :param argument:
    :return:
    """
    A = [[], [], [], []]
    for i in range(len(argument)):
        T = ME_ff(line_vec, param_vec, argument[i])
        for j in range(4):
            A[j].append(T[j])
    return np.concatenate((A[0], A[1], A[2], A[3]))
