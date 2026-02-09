import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

def eta_func(I_val, eta):
    return 1 if I_val > 0 else eta
    

def SOC_func(I, t, z0, Q_Ah_val, eta_val):
    Q = Q_Ah_val * 3600  
    eta_array = np.array([eta_func(i, eta_val) for i in I])
    
    delta_t = np.diff(np.insert(t, 0, t[0]))  
    dz = eta_array * I / Q * delta_t * 100
    z = dz.cumsum() + z0
    
    return z


def SOC_to_OCV(soc_array, df_OCV):
    df_OCV = df_OCV.sort_values("SOC, %")
    soc_data = df_OCV["SOC, %"].values
    U_data = df_OCV["OCV, V"].values

    f = interp1d(soc_data, U_data, bounds_error=False, fill_value=(U_data[0], U_data[-1]))
    
    soc_array = np.array(soc_array)

    return f(soc_array)

def hankel_matrix(signal, s):
    N = len(signal)
    cols = N - s + 1

    H = np.zeros((s, cols))

    for i in range(s):
        H[i, :] = signal[i:i + cols]

    return H

def N4SID(u, y, n, sp, sf, dt):
    '''
    u - управляющее воздействие (I)
    y - измеряемая величина (U)
    sp - глубина Ханкелевой матрицы (past)
    sf - глубина Ханкелевой матрицы (future)
    n - число состояний модели
    dt - шаг дискретизации
    '''

    #---Центрирование данных---
    u = u - np.mean(u)
    y = y - np.mean(y)

    #---Построение Ханкелевых матриц---
    Hu = hankel_matrix(u, sp + sf)
    Hy = hankel_matrix(y, sp + sf)

    Up = Hu[:sp, :]
    Uf = Hu[sp:sp+sf, :]
    Yp = Hy[:sp, :]
    Yf = Hy[sp:sp+sf, :]

    #---Проекция на ортогональное дополнение---
    Z = np.vstack([Up, Yp])
    Q, _ = np.linalg.qr(Z.T, mode='reduced')
    Yf_perp = Yf - Yf @ Q @ Q.T

    #---SVD разложение---
    U_svd, S, Vt = np.linalg.svd(Yf_perp, full_matrices=False)

    #---Обрезка до 3n для уменьшения размерности---
    m = min(5*n, len(S))
    Un = U_svd[:, :m]
    Sn = np.diag(S[:m])
    Vn = Vt[:m, :]

    #---Матрица состояний---
    X = np.sqrt(Sn) @ Vn
    Xk  = X[:, :-1]
    Xk1 = X[:, 1:]
    
    A_hat = Xk1 @ np.linalg.pinv(Xk)

    #---Собственные значения и временные константы---
    eigvals, eigvecs = np.linalg.eig(A_hat)
    eigvals_real = np.real(eigvals)
    eps = 1e-12
    eigvals_proj = np.clip(eigvals_real, eps, 1 - eps)
    tau = -dt / np.log(eigvals_proj)

    #---Отбор n уникальных самых значимых полюсов по τ---
    unique_tau = []
    unique_lambda = []
    eps_tau = 1e-6  # порог для уникальности
    for l, t in zip(eigvals_proj, tau):
        if not any(np.abs(t - ut) < eps_tau for ut in unique_tau):
            unique_tau.append(t)
            unique_lambda.append(l)

    # сортировка по убыванию τ и выбор n
    sorted_idx = np.argsort(-np.array(unique_tau))
    tau_sorted = np.array(unique_tau)[sorted_idx][:n]
    eigvals_sorted = np.array(unique_lambda)[sorted_idx][:n]

    # #---Вывод информации---
    # print("Самые значимые полюса (медленно затухающие первыми):")
    # for i in range(len(tau_sorted)):
    #     print(f"Полюс {i+1}: λ = {eigvals_sorted[i]:.6f}, τ = {tau_sorted[i]:.6f}")

    return tau_sorted


def identify_rc_poles(Uerr, I, n, sp, sf, t):

    y = np.asarray(Uerr)
    u = np.asarray(I)

    t = np.asarray(t)
    dts = np.diff(t)
    dt = np.mean(dts[dts > 0])

    tau = N4SID(u, y, n, sp, sf, dt)
    return tau


def process_dynamic(eta, Q_Ah, t, I, U, OCV, num_poles, sp, sf):

    # --- Subspace Identification (RC) ---

    Uerr = U - OCV
    tau = identify_rc_poles(Uerr, I, num_poles, sp, sf, t)

    # print(f"Идентифицировано RC: {tau}")
    
    # Используем фактическое количество полюсов
    actual_num_poles = len(tau)
    # --- Линейная регрессия для R0 и Rfact ---
    dt = np.diff(t, prepend=t[0])
    Urc = np.zeros((len(I), actual_num_poles))
    
    for k in range(1, len(I)):
        RCfact = np.exp(-dt[k] / tau)
        Urc[k, :] = RCfact * Urc[k-1, :] + (1 - RCfact) * I[k-1]
    
    H = np.column_stack([I, Urc])
    
    W, _, _, _ = np.linalg.lstsq(H, Uerr, rcond=None)
    R0 = W[0]
    Rfact = W[1:]   

    return {'R0': R0, 'R': Rfact, 'tau': tau, 'actual_num_poles': actual_num_poles}





    

def simulate_battery_model(I, t, OCV, eta, Q_Ah, dymanic_parameters):

    R0 = dymanic_parameters['R0']
    R = dymanic_parameters['R']
    tau = dymanic_parameters['tau']
    
    num_poles = len(tau)
    n = len(I)

    # Инициализация напряжений на RC-цепях
    U_rc = np.zeros(num_poles)
    U_sim = np.zeros(n)

    # Разности времени (dt может меняться)
    dt = np.diff(t, prepend=t[0])

    for k in range(n):
        # Коэффициенты экспоненциального затухания
        alpha = np.exp(-dt[k] / tau)

        # Обновляем напряжения на RC-цепях
        U_rc = alpha * U_rc + (1 - alpha) * I[k] * R

        # Напряжение поляризации
        U_polarization = np.sum(U_rc)

        # Клеммное напряжение
        # Поскольку ток > 0 при ЗАРЯДЕ, напряжение должно РАСТИ:
        U_sim[k] = OCV[k] + R0 * I[k] + U_polarization

    return U_sim

def RMSE(U, U_sim):
    return np.sqrt(np.mean((U - U_sim) ** 2))