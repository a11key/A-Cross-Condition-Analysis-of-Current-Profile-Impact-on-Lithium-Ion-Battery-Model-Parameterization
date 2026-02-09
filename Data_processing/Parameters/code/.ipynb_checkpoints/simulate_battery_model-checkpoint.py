import numpy as np
from SOC_Coulomb import SOC_func
from OCV_SOC import soc_to_ocv

def simulate_battery_model(current, model, time, z0, OCV):
    """
    Симуляция батареи на основе идентифицированной модели
    """
    R0 = model['R0']
    R = model['R']
    RC = model['RC']
    Q_Ah = model['Q']
    eta = model['eta']
    
    num_poles = len(RC)
    n = len(current)

    # Инициализация напряжений на RC-цепях
    V_rc = np.zeros(num_poles)
    V_sim = np.zeros(n)

    # Разности времени (dt может меняться)
    dt = np.diff(time, prepend=time[0])
    dt[dt <= 0] = np.median(dt[dt > 0])  # защита от нулевых/отрицательных dt

    for k in range(n):
        # Коэффициенты экспоненциального затухания
        alpha = np.exp(-dt[k] / RC)

        # Обновляем напряжения на RC-цепях
        V_rc = alpha * V_rc + (1 - alpha) * current[k] * R

        # Напряжение поляризации
        V_polarization = np.sum(V_rc)

        # Клеммное напряжение
        # Поскольку ток > 0 при ЗАРЯДЕ, напряжение должно РАСТИ:
        V_sim[k] = OCV[k] + R0 * current[k] + V_polarization

    return V_sim