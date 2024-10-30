import geopandas as gpd
import pandas as pd

def load_grid_data(shp_file_path):
    gdf = gpd.read_file(shp_file_path)
    return gdf[['row_index', 'col_index', 'Elevation', 'Junction']]

def load_flooding_data(flooding_file_path, selected_time):
    flooding_data_raw = pd.read_excel(flooding_file_path, sheet_name='Sheet1')
    flooding_data = flooding_data_raw[flooding_data_raw['Time'] == selected_time].T.drop('Time').reset_index()
    flooding_data.columns = ['junction_id', 'flooding_value']
    return flooding_data

hello