import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# 데이터 불러오기
# SHP 파일 경로 설정
shp_file_path = 'C:/Users/정주영/Desktop/2024-2/종합설계/코드/flooding_test1/DEM_GRID/DEM_GRID.shp'

# SHP 파일 읽기 - 고도와 Junction 정보 가져오기
gdf = gpd.read_file(shp_file_path)
grid_data = gdf[['row_index', 'col_index', 'Elevation', 'Junction']]

# 엑셀 파일 경로 설정
flooding_file_path = 'C:/Users/정주영/Desktop/2024-2/종합설계/코드/flooding_test1/Junction_Flooding_1.xlsx'

# 엑셀 파일 읽기 - 특정 시간대 데이터를 필터링하여 Junction ID와 flooding 값으로 변환
flooding_data_raw = pd.read_excel(flooding_file_path, sheet_name='Sheet1')
selected_time = '2011-07-27 07:10:00'
flooding_data = flooding_data_raw[flooding_data_raw['Time'] == selected_time].T
flooding_data = flooding_data.drop('Time').reset_index()
flooding_data.columns = ['junction_id', 'flooding_value']

# Junction ID를 기준으로 고도 값 및 위치 정보와 병합
grid_data = grid_data.merge(flooding_data, left_on='Junction', right_on='junction_id', how='left')

# Grid 데이터를 64x64 배열 형태로 변환
grid_array = np.zeros((64, 64), dtype=[('elevation', 'f8'), ('junction_id', 'U10'), ('flooding_value', 'f8')])

# 고도 값으로 grid_array 초기화
for _, row in grid_data.iterrows():
    x, y = int(row['col_index']), int(row['row_index'])
    elevation = row['Elevation']
    junction_id = row['Junction']
    flooding_value = row['flooding_value'] if pd.notna(row['flooding_value']) else np.nan
    grid_array[y, x] = (elevation, junction_id, flooding_value)

# 침수 최저점 찾기 함수 (테두리까지 탐색)
def find_inundation_low_point(x, y, grid_array):
    queue = deque([(x, y)])
    visited = set()
    visited.add((x, y))
    lowest_point = (x, y)
    lowest_elevation = grid_array[y, x]['elevation']

    while queue:
        current_x, current_y = queue.popleft()
        current_elevation = grid_array[current_y, current_x]['elevation']
        
        # 인접한 8개의 셀 좌표
        neighbors = [(current_x + dx, current_y + dy) 
                     for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                     if (dx != 0 or dy != 0)]
        
        # 유효한 이웃 셀 필터링
        valid_neighbors = [(nx, ny) for nx, ny in neighbors 
                           if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in visited]
        
        for nx, ny in valid_neighbors:
            neighbor_elevation = grid_array[ny, nx]['elevation']
            
            if neighbor_elevation <= current_elevation:
                queue.append((nx, ny))
                visited.add((nx, ny))
                
                # 최저점 갱신
                if neighbor_elevation < lowest_elevation:
                    lowest_elevation = neighbor_elevation
                    lowest_point = (nx, ny)
    
    return lowest_point

# 각 Junction의 침수 최저점 찾기
inundation_points = []
for _, row in flooding_data.iterrows():
    junction_id = row['junction_id']
    flooding_value = row['flooding_value']
    
    # Junction ID 위치 찾기
    flood_cell = grid_data[grid_data['junction_id'] == junction_id]
    if not flood_cell.empty:
        x, y = int(flood_cell.iloc[0]['col_index']), int(flood_cell.iloc[0]['row_index'])
        
        # 침수 최저점 찾기
        low_point_x, low_point_y = find_inundation_low_point(x, y, grid_array)
        inundation_points.append((junction_id, low_point_x, low_point_y))

# 침수 최저점 출력
print("Inundation low points for each flooding cell:")
for junction_id, lx, ly in inundation_points:
    print(f"Junction ID: {junction_id}, Inundation Low Point: ({lx}, {ly})")

# 그래프 그리기
plt.figure(figsize=(10, 10))

# 고도 배열 생성
elevation_array = grid_array['elevation'].copy()
elevation_array[elevation_array == 999] = -1

# 색상 매핑 설정
cmap = plt.get_cmap('terrain')
norm = plt.Normalize(vmin=-1, vmax=np.max(elevation_array[elevation_array != -1]))

# 고도를 배경으로 표시
plt.imshow(elevation_array, cmap=cmap, norm=norm, origin='lower')

# 침수 최저점을 파란 점으로 표시
for junction_id, lx, ly in inundation_points:
    plt.plot(lx, ly, 'bo', markersize=8)

# y축 반전
plt.gca().invert_yaxis()

# 그래프 제목 및 레이블 설정
plt.title('Flood Inundation Low Points')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Elevation')
plt.show()
