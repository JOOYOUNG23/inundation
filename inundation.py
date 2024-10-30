def find_inundation_low_point(x, y, grid_array):
    current_x, current_y = x, y
    visited = set()
    cells_to_visit = [(current_x, current_y)]

    while cells_to_visit:
        current_x, current_y = cells_to_visit.pop(0)
        visited.add((current_x, current_y))
        
        neighbors = [(current_x + dx, current_y + dy) 
                     for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                     if (dx != 0 or dy != 0)]
        
        valid_neighbors = [(nx, ny) for nx, ny in neighbors 
                           if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in visited]
        
        current_elevation = grid_array[current_y, current_x]['elevation']
        
        lower_or_equal_neighbors = [(nx, ny) for nx, ny in valid_neighbors 
                                   if grid_array[ny, nx]['elevation'] <= current_elevation]
        
        if lower_or_equal_neighbors:
            cells_to_visit.extend([cell for cell in lower_or_equal_neighbors if cell not in cells_to_visit])
        
    return current_x, current_y
