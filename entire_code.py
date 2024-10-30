import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
# SHP 파일 경로 설정
shp_file_path = 'C:/Users/정주영/Desktop/2024-2/종합설계/코드/flooding_test1/DEM_GRID/DEM_GRID.shp'

# SHP 파일 읽기 - 고도와 Junction 정보 가져오기
gdf = gpd.read_file(shp_file_path)
# 올바른 열 이름으로 수정
grid_data = gdf[['row_index', 'col_index', 'Elevation', 'Junction']]

# 엑셀 파일 경로 설정
flooding_file_path = 'C:/Users/정주영/Desktop/2024-2/종합설계/코드/flooding_test1/Junction_Flooding_1.xlsx'

# 엑셀 파일 읽기 - 특정 시간대 데이터를 필터링하여 Junction ID와 flooding 값으로 변환
flooding_data_raw = pd.read_excel(flooding_file_path, sheet_name='Sheet1')

# 특정 시간대 데이터 선택 (예: '2011-07-27 07:10:00')
selected_time = '2011-07-27 07:10:00'
flooding_data = flooding_data_raw[flooding_data_raw['Time'] == selected_time].T  # 전치하여 열과 행 전환

# 데이터프레임 변환 - Time 제외하고 Junction과 flooding 값으로 변환
flooding_data = flooding_data.drop('Time').reset_index()
flooding_data.columns = ['junction_id', 'flooding_value']  # 열 이름 변경

# Junction ID를 기준으로 고도 값 및 위치 정보와 병합
# 'Junction'을 기준으로 merge
grid_data = grid_data.merge(flooding_data, left_on='Junction', right_on='junction_id', how='left')

# Grid 데이터를 64x64 배열 형태로 변환
grid_array = np.zeros((64, 64), dtype=[('elevation', 'f8'), ('junction_id', 'U10'), ('flooding_value', 'f8')])

# 고도 값으로 grid_array 초기화
for _, row in grid_data.iterrows():
    x, y = int(row['col_index']), int(row['row_index'])  # col_index, row_index를 바꾸는 것이 중요함
    elevation = row['Elevation']
    junction_id = row['Junction']
    flooding_value = row['flooding_value'] if pd.notna(row['flooding_value']) else np.nan  # NaN 처리
    grid_array[y, x] = (elevation, junction_id, flooding_value)

# 침수 최저점 찾기 함수
def find_inundation_low_point(x, y, grid_array):
    current_x, current_y = x, y
    visited = set()  # 방문한 셀을 기록
    cells_to_visit = [(current_x, current_y)]  # 탐색할 셀 목록

    while cells_to_visit:
        current_x, current_y = cells_to_visit.pop(0)  # 셀 목록에서 다음 셀 선택
        visited.add((current_x, current_y))  # 현재 셀 방문 기록
        
        # # J19의 경우, 탐색 좌표를 출력
        # if junction_id == 'J19':
        #     print(f"Exploring: ({current_x}, {current_y}), Elevation: {grid_array[current_y, current_x]['elevation']}")

        # 인접한 8개의 셀의 좌표
        neighbors = [(current_x + dx, current_y + dy) 
                     for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                     if (dx != 0 or dy != 0)]
        
        # 유효한 이웃 셀 필터링 (그리드 바깥 제외)
        valid_neighbors = [(nx, ny) for nx, ny in neighbors 
                           if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in visited]
        
        # 현재 셀의 고도
        current_elevation = grid_array[current_y, current_x]['elevation']
        
        # 고도가 낮거나 같은 이웃 셀 찾기
        lower_or_equal_neighbors = [(nx, ny) for nx, ny in valid_neighbors 
                                   if grid_array[ny, nx]['elevation'] <= current_elevation]
        
        if lower_or_equal_neighbors:
            # 모든 고도가 낮거나 같은 셀을 포함
            for nx, ny in lower_or_equal_neighbors:
                # 해당 셀이 방문한 적이 없으면 추가
                if (nx, ny) not in visited and (nx, ny) not in cells_to_visit:
                    cells_to_visit.append((nx, ny))
        
        # 더 이상 탐색할 셀이 없을 경우 현재 좌표 반환
    return current_x, current_y

# 각 Junction의 침수 최저점 찾기
inundation_points = []
for _, row in flooding_data.iterrows():
    junction_id = row['junction_id']
    flooding_value = row['flooding_value']
    
    # Junction ID 위치 찾기
    flood_cell = grid_data[grid_data['junction_id'] == junction_id]
    if not flood_cell.empty:
        x, y = int(flood_cell.iloc[0]['col_index']), int(flood_cell.iloc[0]['row_index'])  # col_index, row_index
        
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

# 고도가 999인 셀을 -1로 설정 (검은색 표시를 위해)
elevation_array[elevation_array == 999] = -1

# 색상 매핑을 위한 고도 범위 설정
cmap = plt.get_cmap('terrain')
norm = plt.Normalize(vmin=-1, vmax=np.max(elevation_array[elevation_array != -1]))

# 고도를 배경으로 표시 (999인 셀은 검은색으로 처리)
plt.imshow(elevation_array, cmap=cmap, norm=norm, origin='lower')

# 침수 최저점을 파란 점으로 표시
for junction_id, lx, ly in inundation_points:
    plt.plot(lx, ly, 'bo', markersize=8)  # 침수 최저점 표시

# y축 반전
plt.gca().invert_yaxis()

# 그래프 제목 및 레이블 설정
plt.title('Flood Inundation Low Points')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Elevation')
plt.show()  # 그래프 출력
