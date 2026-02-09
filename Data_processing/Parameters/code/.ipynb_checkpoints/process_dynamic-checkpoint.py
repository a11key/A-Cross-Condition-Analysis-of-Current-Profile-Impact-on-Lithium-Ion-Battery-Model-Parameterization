import numpy as np

def hankel_matrix(signal, s):
    N = len(signal)
    cols = N - s + 1

    H = np.zeros((s, cols))

    for i in range(s):
        H[i, :] = signal[i:i + cols]

    return H

# def N4SID(u, y, n, sp, sf, dt):
#     '''
#     u - управляющее воздействие (I)
#     y - измеряемая величина (U)
#     sp - глубина Ханкелевой матрицы
#     sf - глубина Ханкелевой матрицы
#     '''

#     #---Центрирование данных (избавление от шумов)---
#     u = u - np.mean(u)
#     y = y - np.mean(y)

#     #---Построение Ханкелевых матриц---
#     Hu = hankel_matrix(u, sp + sf)
#     Hy = hankel_matrix(y, sp + sf)

#     Up = Hu[:sp, :]
#     Uf = Hu[sp:sp+sf, :]

#     Yp = Hy[:sp, :]
#     Yf = Hy[sp:sp+sf, :]

#     #---Проекция на ортогональное дополнение---

#     Z = np.vstack([Up, Yp])

#     Q, _ = np.linalg.qr(Z.T, mode='reduced')
#     Yf_perp = Yf - Yf @ Q @ Q.T

#     #---SVD разложение---
#     U, S, Vt = np.linalg.svd(Yf_perp, full_matrices=False)

#     #---Выбор n полюсов---
#     Un = U[:, :n]                     # (sf x n)
#     Sn = np.diag(S[:n])               # (n x n)
#     Vn = Vt[:n, :]                    # (n x Nc)

#     X = np.sqrt(Sn) @ Vn              # (n x Nc)

#     Xk  = X[:, :-1]                   # (n x (Nc-1))
#     Xk1 = X[:, 1:]                    # (n x (Nc-1))
    
#     A_hat = Xk1 @ np.linalg.pinv(Xk)  # (n x n)

#     eigvals, eigvecs = np.linalg.eig(A_hat)

#     eigvals_real = np.real(eigvals)

#     eps = 1e-12
#     eigvals_proj = np.clip(eigvals_real, eps, 1 - eps)

#     A_phys = eigvecs @ np.diag(eigvals_proj) @ np.linalg.inv(eigvecs)

#     tau = -dt / np.log(eigvals_proj)

#     return tau

def N4SID(u, y, n, sp, sf, dt):

    u = u - np.mean(u)
    y = y - np.mean(y)

    Hu = hankel_matrix(u, sp + sf)
    Hy = hankel_matrix(y, sp + sf)

    Up = Hu[:sp, :]
    Yp = Hy[:sp, :]
    Yf = Hy[sp:sp+sf, :]

    Z = np.vstack([Up, Yp])
    Q, _ = np.linalg.qr(Z.T, mode='reduced')
    Yf_perp = Yf - Yf @ Q @ Q.T

    U, S, Vt = np.linalg.svd(Yf_perp, full_matrices=False)

    # ---- НЕ режем до n ----
    S_norm = S / S[0]
    r = np.sum(S_norm > 1e-2)

    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = Vt[:r, :]

    X = np.sqrt(Sr) @ Vr

    Xk  = X[:, :-1]
    Xk1 = X[:, 1:]

    A_hat = Xk1 @ np.linalg.pinv(Xk)

    eigvals = np.linalg.eigvals(A_hat)
    eigvals = eigvals[np.isreal(eigvals)].real
    eigvals = eigvals[(eigvals > 0) & (eigvals < 1)]

    tau_all = -dt / np.log(eigvals)

    tau_all = np.sort(tau_all)
    tau_all = unique_taus(tau_all)

    return tau_all[:n]


def identify_rc_poles(verr, etaik, n, sp, sf, time):
    """
    Подпространственная идентификация RC-постоянных через (N4SID)
    С учётом переменного временного шага dt по данным.
    """

    y = np.asarray(verr)
    u = np.asarray(etaik)

    # --- Расчёт среднего dt ---

    time = np.asarray(time)
    dts = np.diff(time)
    dt = np.mean(dts[dts > 0])  # средний положительный шаг

    tau = N4SID(u, y, n, sp, sf, dt)
    return tau


def process_dynamic(data, model, num_poles, sp, sf):

    eta = model['eta']
    Q = model['Q']

    print(f"⚙️ η = {eta:.3f}, Q = {Q:.3f} Ah")

    # --- Расчёт SOC ---
    time = np.array(data['time'])
    current = np.array(data['current'])
    voltage = np.array(data['voltage'])
    etaik = current
    
    
    OCV = model['OCV_ref']

    # --- Subspace Identification (RC) ---
    vk = voltage
    verr = vk - OCV
    RC = identify_rc_poles(verr, etaik, num_poles, sp, sf, time)

    print(f"🔍 Идентифицировано RC: {RC}")
    
    # Используем фактическое количество полюсов
    actual_num_poles = len(RC)
    # --- Линейная регрессия для R0 и Rfact ---
    dt = np.diff(time, prepend=time[0])
    vrc = np.zeros((len(etaik), actual_num_poles))
    
    for k in range(1, len(etaik)):
        RCfact = np.exp(-dt[k] / RC)
        vrc[k, :] = RCfact * vrc[k-1, :] + (1 - RCfact) * etaik[k-1]
    
    # ✅ знак физически согласован
    H = np.column_stack([etaik, vrc])
    
    W, _, _, _ = np.linalg.lstsq(H, verr, rcond=None)
    R0 = W[0]
    Rfact = W[1:]

    model.update({
        'R0': R0,
        'R': Rfact,
        'RC': RC,
        'actual_num_poles': actual_num_poles
    })

    print("✅ Идентификация завершена:")
    print(f"  R0 = {R0:.6f} Ом")
    print(f"  RC = {RC} секунд")
    print(f"  R = {Rfact} Ом")
    print(f"  Фактическое количество полюсов: {actual_num_poles}")

    return model