import numpy as np

# --- Функция эффективности ---
def eta_func(I_val, eta):
    return 1 if I_val > 0 else eta
    
# --- Функция расчета SOC ---
def SOC_func(I, t, z0, Q_Ah_val, eta_val):
    """
    I : pd.Series или np.array, ток [A]
    t : pd.Series или np.array, время [s]
    z0 : float, начальный SOC
    Q_Ah : емкость батареи в Ah
    eta_val : эффективность разряда (0 < eta_val <= 1)
    """
    Q = Q_Ah_val * 3600  # Ah -> Coulomb
    # Коэффициент эффективности для каждого тока
    eta_array = np.array([eta_func(i, eta_val) for i in I])
    
    delta_t = np.diff(np.insert(t, 0, t[0]))  # шаги времени
    dz = eta_array * I / Q * delta_t
    z = dz.cumsum() + z0
    z = np.clip(z, 0, 1)  # ограничиваем SOC [0,1]
    return z