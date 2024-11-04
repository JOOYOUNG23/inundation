import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# 데이터 불러오기
shp_file_path = 'C:/Users/정주영/Desktop/2024-2/종합설계/코드/flooding_test1/DEM_GRID/DEM_GRID.shp'
gdf = gpd.read_file(shp_file_path)
grid_data = gdf[['row_index', 'col_index', 'Elevation', 'Junction']]

flooding_file_path = 'C:/Users/정주영/Desktop/2024-2/종합설계/코드/flooding_test1/Junction_Flooding_1.xlsx'
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

# 침수 최저점 찾기 함수
def find_inundation_low_points(x, y, grid_array):
    queue = deque([(x, y)])
    visited = set()
    visited.add((x, y))
    lowest_points = [(x, y)]
    lowest_elevation = grid_array[y, x]['elevation']

    while queue:
        current_x, current_y = queue.popleft()
        current_elevation = grid_array[current_y, current_x]['elevation']
        
        # 인접한 8개의 셀 좌표
        neighbors = [(current_x + dx, current_y + dy) 
                     for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                     if (dx != 0 or dy != 0)]
        
        for nx, ny in neighbors:
            if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in visited:
                visited.add((nx, ny))
                neighbor_elevation = grid_array[ny, nx]['elevation']
                
                if neighbor_elevation < lowest_elevation:
                    lowest_points = [(nx, ny)]
                    lowest_elevation = neighbor_elevation
                elif neighbor_elevation == lowest_elevation:
                    lowest_points.append((nx, ny))
                    
                if neighbor_elevation <= current_elevation:
                    queue.append((nx, ny))
    
    return lowest_points, lowest_elevation

# 같은 고도의 셀 탐색 함수
def find_connected_same_elevation_cells(x, y, elevation, grid_array):
    queue = deque([(x, y)])
    visited = set()
    visited.add((x, y))
    connected_cells = [(x, y)]

    while queue:
        current_x, current_y = queue.popleft()
        
        # 인접한 8개의 셀 좌표
        neighbors = [(current_x + dx, current_y + dy) 
                     for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                     if (dx != 0 or dy != 0)]
        
        for nx, ny in neighbors:
            if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in visited:
                if grid_array[ny, nx]['elevation'] == elevation:
                    queue.append((nx, ny))
                    visited.add((nx, ny))
                    connected_cells.append((nx, ny))

    return connected_cells

# 각 Junction의 침수 최저점 찾기 및 초기침수범위 설정
initial_flooded_cells = []
lowest_elevation = float('inf')

for _, row in flooding_data.iterrows():
    junction_id = row['junction_id']
    flooding_value = row['flooding_value']
    
    # Junction ID 위치 찾기
    flood_cell = grid_data[grid_data['junction_id'] == junction_id]
    if not flood_cell.empty:
        x, y = int(flood_cell.iloc[0]['col_index']), int(flood_cell.iloc[0]['row_index'])
        
        # 침수 최저점 찾기
        low_points, elevation = find_inundation_low_points(x, y, grid_array)
        
        # 초기 침수 범위 설정
        for low_x, low_y in low_points:
            initial_flooded_cells.extend(find_connected_same_elevation_cells(low_x, low_y, elevation, grid_array))

# 초기 침수 범위의 고도 및 셀 영역
lowest_elevation = min(grid_array[ly, lx]['elevation'] for lx, ly in initial_flooded_cells if grid_array[ly, lx]['elevation'] != 999)
cell_area = 244.1406  # 각 셀의 면적

# 초기 H 계산
def calculate_initial_H(flooded_cells, lowest_elevation, total_flooding, cell_area):
    flooded_cells_count = len(flooded_cells)
    if flooded_cells_count == 0:
        return 0  # flooded_cells가 없으면 H는 0
    H = (total_flooding / (cell_area * flooded_cells_count)) + lowest_elevation
    return H

# 총 침수량
total_flooding = sum(row['flooding_value'] for _, row in flooding_data.iterrows() if pd.notna(row['flooding_value']))+1000
# 침수 범위 초기화
flooded_cells = set(initial_flooded_cells)

# 초기 H 계산
H = calculate_initial_H(flooded_cells, lowest_elevation, total_flooding, cell_area)

# 시뮬레이션 진행
while True:
    new_flooded_cells = set(flooded_cells)  # 현재 flooded_cells 복사본을 생성
    max_depth = float('inf')  # 최대 수심 초기화
    max_depth_cells = []  # 최대 수심을 가진 셀 저장
    
    # 인접 셀을 순회하여 침수 범위를 확장
    for x, y in flooded_cells:
        neighbors = [(x + dx, y + dy) 
                     for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                     if (dx != 0 or dy != 0)]
        
        # 현재 셀의 고도
        current_cell_elevation = grid_array[y, x]['elevation']
        
        # 인접한 셀 중에서 현재 셀보다 높은 고도를 찾기
        higher_adjacent_elevations = []

        for nx, ny in neighbors:
            if 0 <= nx < 64 and 0 <= ny < 64:
                if (nx, ny) not in flooded_cells: # 이미 침수된 영역은 탐색에서 제외
                    adjacent_elevation = grid_array[ny, nx]['elevation']
                    if adjacent_elevation != 999:  # 고도 999 제외
                        higher_adjacent_elevations.append(adjacent_elevation)
            
        # 두 번째로 낮은 고도 찾기
        if len(higher_adjacent_elevations) >= 1:  # 현재 셀보다 높은 셀이 최소 하나 존재해야 함
            second_lowest_elevation = min(higher_adjacent_elevations)  # 가장 낮은 고도
            max_depth = second_lowest_elevation  # 최대 수심

            # H와 최대 수심 비교
            if H >= max_depth:
                for nx, ny in neighbors:
                    if 0 <= nx < 64 and 0 <= ny < 64:
                        if (nx, ny) not in new_flooded_cells and grid_array[ny, nx]['elevation'] != 999:
                            # 같은 고도인 셀까지 추가
                            connected_cells = find_connected_same_elevation_cells(nx, ny, grid_array[ny, nx]['elevation'], grid_array)
                            new_flooded_cells.update(connected_cells)

    flooded_cells = new_flooded_cells  # 업데이트된 flooded_cells로 변경

    # H 업데이트
    new_H = (total_flooding / (cell_area * len(flooded_cells))) + lowest_elevation

    H = new_H  # H를 업데이트

    if H < max_depth:  # H가 최대 수심보다 작으면 종료
        break

# 그래프 그리기
plt.figure(figsize=(10, 10))

# 고도 배열 생성
elevation_array = grid_array['elevation'].copy()
elevation_array[elevation_array == 999] = -1  # 고도 999를 -1로 변환

# 색상 매핑 설정
cmap = plt.get_cmap('terrain')
norm = plt.Normalize(vmin=-1, vmax=np.max(elevation_array[elevation_array != -1]))

# 고도를 배경으로 표시
plt.imshow(elevation_array, cmap=cmap, norm=norm, origin='lower')

# 침수 범위를 파란색으로 표시
for cx, cy in flooded_cells:
    plt.plot(cx, cy, 'bo', markersize=5)

# y축 반전
plt.gca().invert_yaxis()

# 그래프 제목 및 레이블
plt.title('Flooded Areas and Elevation Map')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()
