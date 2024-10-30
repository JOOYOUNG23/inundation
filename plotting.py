import matplotlib.pyplot as plt
import numpy as np

def plot_inundation(grid_array, inundation_points):
    elevation_array = grid_array['elevation'].copy()
    elevation_array[elevation_array == 999] = -1
    cmap = plt.get_cmap('terrain')
    norm = plt.Normalize(vmin=-1, vmax=np.max(elevation_array[elevation_array != -1]))

    plt.imshow(elevation_array, cmap=cmap, norm=norm, origin='lower')
    for junction_id, lx, ly in inundation_points:
        plt.plot(lx, ly, 'bo', markersize=8)
    
    plt.gca().invert_yaxis()
    plt.title('Flood Inundation Low Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Elevation')
    plt.show()
