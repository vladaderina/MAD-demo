import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('output.csv', index_col='timestamp', parse_dates=True)

# Тест Дики-Фуллера
result_adf = adfuller(data['value'])
print('ADF Statistic:', result_adf[0])
print('p-value:', result_adf[1])
print('Critical Values:', result_adf[4])

# Тест KPSS
result_kpss = kpss(data['value'])
print('KPSS Statistic:', result_kpss[0])
print('p-value:', result_kpss[1])
print('Critical Values:', result_kpss[3])
