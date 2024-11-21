import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import deque

# 데이터 불러오기
shp_file_path = 'C:/Users/정주영/Desktop/2024-2/종합설계/코드/flooding_test1/DEM_GRID/DEM_GRID.shp'
gdf = gpd.read_file(shp_file_path)
grid_data = gdf[['row_index', 'col_index', 'Elevation', 'Junction']]

flooding_file_path = 'C:/Users/정주영/Desktop/2024-2/종합설계/코드/flooding_test1/Junction_Flooding_1.xlsx'
flooding_data_raw = pd.read_excel(flooding_file_path, sheet_name='Sheet1')
selected_time = '2011-07-27 08:40:00'
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
total_flooding = 0
total_flooding = sum(row['flooding_value'] for _, row in flooding_data.iterrows() if pd.notna(row['flooding_value']))
total_flooding = total_flooding * 600

# 침수 범위 초기화
flooded_cells = set(initial_flooded_cells)

# 초기 H 계산
H = calculate_initial_H(flooded_cells, lowest_elevation, total_flooding, cell_area)

# 고도 별 총 유출량 계산 (total_flooding_computed)
def compute_total_flooding(H, elevation_groups, cell_area):
    total_flooding_computed = 0
    for elevation, cells in elevation_groups.items():
        flooded_cells_count = len(cells)
        total_flooding_computed += (H - elevation) * cell_area * flooded_cells_count
    return total_flooding_computed

# 이분법을 사용하여 total_flooding에 맞는 최적의 H를 찾는 함수.
def find_optimal_H(total_flooding, elevation_groups, cell_area, H_min, H_max, tolerance=1e-5):    
    while H_max - H_min > tolerance:  # tolerance는 원하는 정확도
        H_mid = (H_min + H_max) / 2
        total_flooding_computed = compute_total_flooding(H_mid, elevation_groups, cell_area)
        
        if total_flooding_computed < total_flooding:
            H_min = H_mid  # computed flooding이 부족하면 H를 늘린다
        else:
            H_max = H_mid  # computed flooding이 넘치면 H를 줄인다

    return (H_min + H_max) / 2  # 최종적으로 H 값을 반환

# 시뮬레이션 진행
while True:
    new_flooded_cells = set(flooded_cells)  # 현재 flooded_cells 복사본을 생성
    max_depth = float('inf')  # 최대 수심 초기화

    # 침수된 셀의 모든 인접 셀의 고도를 수집하기 위해 사용될 집합
    all_higher_adjacent_elevations = set()

    # 인접 셀을 순회하여 침수 범위를 확장
    for x, y in flooded_cells:
        neighbors = [(x + dx, y + dy) 
                     for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                     if (dx != 0 or dy != 0)]
        
        for nx, ny in neighbors:
            if 0 <= nx < 64 and 0 <= ny < 64:
                if (nx, ny) not in flooded_cells:  # 이미 침수된 영역은 탐색에서 제외
                    adjacent_elevation = grid_array[ny, nx]['elevation']
                    if adjacent_elevation != 999:  # 고도 999 제외
                        all_higher_adjacent_elevations.add(adjacent_elevation)

    # 두 번째로 낮은 고도 찾기
    if len(all_higher_adjacent_elevations) >= 1:  # 현재 셀보다 높은 셀이 최소 하나 존재해야 함
        second_lowest_elevation = min(all_higher_adjacent_elevations)  # 가장 낮은 고도
        max_depth = second_lowest_elevation  # 최대 수심

        # H와 최대 수심 비교
        if H >= max_depth:
            # 인접 셀 중 두 번째로 낮은 고도를 가진 셀 찾기
            lowest_neighbor_cells = [(nx, ny) for x, y in flooded_cells 
                                     for nx, ny in [(x + dx, y + dy) 
                                                    for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                                                    if (dx != 0 or dy != 0)]
                                     if 0 <= nx < 64 and 0 <= ny < 64 
                                     and grid_array[ny, nx]['elevation'] == second_lowest_elevation]

            for nx, ny in lowest_neighbor_cells:
                if (nx, ny) not in new_flooded_cells and grid_array[ny, nx]['elevation'] != 999:
                    # 같은 고도인 셀까지 추가
                    connected_cells = find_connected_same_elevation_cells(nx, ny, grid_array[ny, nx]['elevation'], grid_array)
                    new_flooded_cells.update(connected_cells)

    flooded_cells = new_flooded_cells  # 업데이트된 flooded_cells로 변경

    # H 계산을 위해 고도별로 침수 셀을 그룹화
    elevation_groups = {}
    for x, y in flooded_cells:
        cell_elevation = grid_array[y, x]['elevation']
        if cell_elevation != 999:
            if cell_elevation not in elevation_groups:
                elevation_groups[cell_elevation] = []
            elevation_groups[cell_elevation].append((x, y))

    # H 값을 이분법을 사용하여 최적화
    H_min = lowest_elevation  # H의 최소값은 최저 고도
    H_max = 41.68772125  # H의 최대값은 41.68772125
    H = find_optimal_H(total_flooding, elevation_groups, cell_area, H_min, H_max)

    # 종료 조건: H가 max_depth보다 작으면 종료
    if H < max_depth:
        break

# 그래프 그리기
plt.figure(figsize=(10, 10))

# 고도 배열 생성
elevation_array = grid_array['elevation'].copy()
elevation_array[elevation_array == 999] = np.nan  # 고도 999를 NaN으로 변환

# 'terrain' 컬러맵 복사
cmap = mpl.cm.get_cmap("terrain").copy()
cmap.set_bad(color='black')  # NaN 값을 까만색으로 표시
norm = plt.Normalize(vmin=-1, vmax=np.nanmax(elevation_array))  # NaN 제외한 최대값 계산

# 고도를 배경으로 표시
plt.imshow(elevation_array, cmap=cmap, norm=norm, origin='lower')

# flooded_cells의 각 셀에 대해 inundation_H에 따라 색상 결정
min_inundation_H = float('inf')  # inundation_H의 최소값
max_inundation_H = float('-inf')  # inundation_H의 최대값

# inundation_H 값의 최소값과 최대값을 찾기
for cx, cy in flooded_cells:
    cell_elevation = grid_array[cy, cx]['elevation']
    inundation_H = H - cell_elevation
    min_inundation_H = min(min_inundation_H, inundation_H)
    max_inundation_H = max(max_inundation_H, inundation_H)

# 정규화된 inundation_H 값을 계산하고 색상 결정
for cx, cy in flooded_cells:
    cell_elevation = grid_array[cy, cx]['elevation']
    inundation_H = H - cell_elevation

    # inundation_H 값을 정규화 (0~1 범위로)
    normalized_inundation_H = (inundation_H - min_inundation_H) / (max_inundation_H - min_inundation_H)

    # 정규화된 inundation_H 값에 따라 색상 결정
    if normalized_inundation_H <= 0.2:
        color = 'cornflowerblue'  # 0.2 이하
    elif 0.2 < normalized_inundation_H <= 0.3:
        color = 'royalblue'  # 0.2 ~ 0.3
    elif 0.3 < normalized_inundation_H <= 0.5:
        color = 'mediumblue'  # 0.3 ~ 0.5
    else:
        color = 'darkblue'  # 0.5 이상

    # 해당 셀에 색상 적용하여 점 그리기
    plt.plot(cx, cy, 's', markersize=5, color=color)

# y축 반전
plt.gca().invert_yaxis()

# 그래프 제목 및 레이블
plt.title('Flooded Areas')

# 색상 범례 추가
from matplotlib.patches import Patch

# 범례 항목을 위한 색상 지정
legend_elements = [
    Patch(color='cornflowerblue', label='0 ~ 0.2m'),
    Patch(color='royalblue', label='0.2 ~ 0.3m'),
    Patch(color='mediumblue', label='0.3 ~ 0.5m'),
    Patch(color='darkblue', label='0.5m ~')
]

# 범례 표시
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

# 그래프 표시
plt.show()