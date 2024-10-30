from data_loading import load_grid_data, load_flooding_data
from grid_initialization import initialize_grid_array
from inundation import find_inundation_low_point
from plotting import plot_inundation

# 파일 경로 및 시간 설정
shp_file_path = 'C:/Users/정주영/Desktop/2024-2/종합설계/코드/flooding_test1/DEM_GRID/DEM_GRID.shp'
flooding_file_path = 'C:/Users/정주영/Desktop/2024-2/종합설계/코드/flooding_test1/Junction_Flooding_1.xlsx'
selected_time = '2011-07-27 07:10:00'

# 데이터 로드
grid_data = load_grid_data(shp_file_path)
flooding_data = load_flooding_data(flooding_file_path, selected_time)

# 그리드 초기화
grid_data = grid_data.merge(flooding_data, left_on='Junction', right_on='junction_id', how='left')
grid_array = initialize_grid_array(grid_data)

# 침수 최저점 찾기
inundation_points = []
for _, row in flooding_data.iterrows():
    junction_id = row['junction_id']
    flood_cell = grid_data[grid_data['junction_id'] == junction_id]
    if not flood_cell.empty:
        x, y = int(flood_cell.iloc[0]['col_index']), int(flood_cell.iloc[0]['row_index'])
        low_point_x, low_point_y = find_inundation_low_point(x, y, grid_array)
        inundation_points.append((junction_id, low_point_x, low_point_y))

# 그래프 출력
plot_inundation(grid_array, inundation_points)
