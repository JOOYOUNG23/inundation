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

# 침수 영역을 찾는 함수
def find_flood_region(x, y, elevation, grid_array):
    queue = deque([(x, y)])
    visited = set([(x, y)])
    flood_region = [(x, y)]  # 최저점과 연결된 동일 고도의 셀들
    
    while queue:
        current_x, current_y = queue.popleft()
        
        # 인접한 8개의 셀 좌표
        neighbors = [(current_x + dx, current_y + dy) 
                     for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                     if (dx != 0 or dy != 0)]
        
        # 유효한 이웃 셀 필터링
        valid_neighbors = [(nx, ny) for nx, ny in neighbors 
                           if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in visited]
        
        for nx, ny in valid_neighbors:
            neighbor_elevation = grid_array[ny, nx]['elevation']
            
            # 최저점과 동일한 고도의 셀을 찾고, 연결된 셀들을 flood_region에 추가
            if neighbor_elevation == elevation:
                queue.append((nx, ny))
                visited.add((nx, ny))
                flood_region.append((nx, ny))
    
    return flood_region

# 침수 영역 확장 함수
def expand_flood_region(flood_region, grid_array):
    # 현재 침수 영역의 테두리 셀 찾기
    boundary_cells = set()
    for x, y in flood_region:
        neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx != 0 or dy != 0)]
        for nx, ny in neighbors:
            if (nx, ny) not in flood_region and 0 <= nx < 64 and 0 <= ny < 64:
                boundary_cells.add((x, y))
                break

    # 테두리 셀의 인접 셀 중에서 가장 낮은 고도를 가진 셀 찾기
    lowest_neighbor = None
    lowest_elevation = float('inf')
    for x, y in boundary_cells:
        neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx != 0 or dy != 0)]
        for nx, ny in neighbors:
            if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in flood_region:
                neighbor_elevation = grid_array[ny, nx]['elevation']
                if neighbor_elevation < lowest_elevation:
                    lowest_elevation = neighbor_elevation
                    lowest_neighbor = (nx, ny)

    # 가장 낮은 셀을 시작점으로 새로운 침수 영역 확장
    if lowest_neighbor is not None:
        new_region = find_flood_region(lowest_neighbor[0], lowest_neighbor[1], lowest_elevation, grid_array)
        flood_region.extend(new_region)

    return flood_region

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

# 침수 최저점과 침수 영역을 찾고 시각화
plt.figure(figsize=(10, 10))

# 고도 배열 생성
elevation_array = grid_array['elevation'].copy()
elevation_array[elevation_array == 999] = -1

# 색상 매핑 설정
cmap = plt.get_cmap('terrain')
norm = plt.Normalize(vmin=-1, vmax=np.max(elevation_array[elevation_array != -1]))

# 고도를 배경으로 표시
plt.imshow(elevation_array, cmap=cmap, norm=norm, origin='lower')

# 각 Junction에 대해 침수 최저점과 침수 영역 확장 수행
for junction_id, lx, ly in inundation_points:
    # 최저점의 고도 가져오기
    low_point_elevation = grid_array[ly, lx]['elevation']
    
    # 초기 침수 영역 찾기
    flood_region = find_flood_region(lx, ly, low_point_elevation, grid_array)
    
    # 침수 영역 확장
    expanded_flood_region = expand_flood_region(flood_region, grid_array)
    
    # 확장된 침수 영역을 파란색으로 표시
    for fx, fy in expanded_flood_region:
        plt.plot(fx, fy, 'bo', markersize=4)

# y축 반전
plt.gca().invert_yaxis()

# 그래프 제목 및 레이블 설정
plt.title('Expanded Flood Inundation Regions')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Elevation')
plt.show()
