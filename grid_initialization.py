import numpy as np
import pandas as pd

def initialize_grid_array(grid_data, shape=(64, 64)):
    grid_array = np.zeros(shape, dtype=[('elevation', 'f8'), ('junction_id', 'U10'), ('flooding_value', 'f8')])
    for _, row in grid_data.iterrows():
        x, y = int(row['col_index']), int(row['row_index'])
        elevation = row['Elevation']
        junction_id = row['Junction']
        flooding_value = row['flooding_value'] if pd.notna(row['flooding_value']) else np.nan
        grid_array[y, x] = (elevation, junction_id, flooding_value)
    return grid_array
