# benchmark.py

import time
import numpy as np
import pandas as pd
from timeseries_utils import *

window = 50
series = generate_time_series(100000)

results = []

# NumPy
start = time.time()
ma_numpy = moving_average_numpy(series, window)
end = time.time()
results.append(['NumPy', len(ma_numpy), end-start])

# Pandas
start = time.time()
ma_pandas = moving_average_pandas(pd.Series(series), window)
end = time.time()
results.append(['Pandas', len(ma_pandas), end-start])

# NumExpr
start = time.time()
ma_numexpr = moving_average_numexpr(series, window)
end = time.time()
results.append(['NumExpr', len(ma_numexpr), end-start])

# Numba
start = time.time()
ma_numba = moving_average_numba(series, window)
end = time.time()
results.append(['Numba', len(ma_numba), end-start])

# Save to CSV
df = pd.DataFrame(results, columns=['Method', 'Output Length', 'Execution Time (s)'])
df.to_csv('benchmark_results.csv', index=False)
print(df)
