import numpy as np
from path import DATA_PATH
from analyze_data import npy_vertices_files, npy_faces_files, npy_groups_files
import matplotlib.pyplot as plt
from functions.plotAF_plotMesh_Juan import plotMesh_Juan

# Normalize vertices into a cube with min coordinate target_min and max coordinate target_max
def normalize_vertices(vertices):
    # Calculate the mean and standard deviation
    mean = np.mean(vertices, axis=0)
    std = np.std(vertices, axis=0)
    
    # Avoid division by zero if std is zero
    std[std == 0] = 1.0

    # Normalize the vertices
    normalized_vertices = (vertices - mean) / std
    
    success = check_normalization(normalized_vertices)
    
    if success: 
        return normalized_vertices
    else: 
        print("error with normalization")
    
def check_normalization(vertices):
    """
    Check if the vertices have a mean of 0 and a standard deviation of 1.

    Parameters:
    - vertices: numpy array of shape (n, 3).

    Returns result that states whether or not the vertices have actually been normalized (i.e. mean = 0 and std = 1)
    """
    # Calculate mean and std
    mean = np.mean(vertices, axis=0)
    std = np.std(vertices, axis=0)
    
    # Return value
    result = False
    
    # Return true if all the points in the mean and std array are normalized with a tolerance of .2
    if np.all(np.abs(mean - 0) <= 0.2) and np.all(np.abs(std - 1) <= 0.2): 
        result = True
    else: 
        print("NORMALIZATION FAILED!")
        print("mean")
        print(mean)
        print("std")
        print(std)
    return result

if __name__ == "__main__":
    # Example usage
    # Construct the full path to the .npy file
    npy_file_path = DATA_PATH + "/" + npy_vertices_files[10]
    npy_face = DATA_PATH + "/" + npy_faces_files[10]
    npy_group = DATA_PATH + "/" + npy_groups_files[10]
    
    # Load data
    vertices = np.load(npy_file_path)
    faces = np.load(npy_face)
    groups = np.load(npy_group)
    
    print("NORMALIZATION INTO CUBE")
    normalized_vertices = normalize_vertices(vertices)
    print("VERTICES BEFORE")
    print(vertices)
    print("VERTICES AFTER")
    print(normalized_vertices)
    print("min value after normalizing")
    print(np.min(normalized_vertices, axis=0))
    print("max value after normalizing")
    print(np.max(normalized_vertices, axis=0))
    
    
    # Save plot after preprocessing
    plotMesh_Juan(
            normalized_vertices, faces, groups,
            clim_limits=[0, 11], intensity_mode='vertex',
            color_scheme='Turbo',
            representation='groups_points',
            save_images=True, save_figure_path="out/afterpreprocessing.html"
        )
    
    # Save plot before preprocessing 
    plotMesh_Juan(
            vertices, faces, groups,
            clim_limits=[0, 11], intensity_mode='vertex',
            color_scheme='Turbo',
            representation='groups_points',
            save_images=True, save_figure_path="out/beforepreprocessing.html"
        )
    
    print("END OF NORMALIZATION")