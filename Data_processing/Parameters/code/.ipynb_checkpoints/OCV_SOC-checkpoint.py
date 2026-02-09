import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def soc_to_ocv(csv_path, soc_array, kind='linear'):
    # Загружаем CSV
    df = pd.read_csv(csv_path)
    
    # Получаем массивы
    SOC_data = np.array(df["SOC"])
    U_data = np.array(df["OCV,V"])
    
    # Интерполяция
    f = interp1d(SOC_data, U_data, kind=kind, bounds_error=False, fill_value=(U_data[0], U_data[-1]))
    
    # Возвращаем OCV для заданных SOC
    return f(soc_array)

def ocv_to_soc(csv_path, ocv_array, kind='linear'):
    
    # Загружаем CSV
    df = pd.read_csv(csv_path)
    
    # Проверяем наличие нужных колонок
    if "SOC" not in df.columns or "OCV,V" not in df.columns:
        raise ValueError("CSV должен содержать колонки 'SOC' и 'OCV,V'")
    
    # Получаем массивы
    SOC_data = np.array(df["SOC"])
    U_data = np.array(df["OCV,V"])
    
    # Проверка на монотонность
    if not (np.all(np.diff(U_data) > 0) or np.all(np.diff(U_data) < 0)):
        sort_idx = np.argsort(U_data)
        U_data = U_data[sort_idx]
        SOC_data = SOC_data[sort_idx]
    
    # Интерполяция OCV → SOC
    f = interp1d(U_data, SOC_data, kind=kind, bounds_error=False, fill_value=(SOC_data[0], SOC_data[-1]))
    
    # Возвращаем SOC для заданных OCV
    return f(ocv_array)

