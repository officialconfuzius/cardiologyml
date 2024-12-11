import numpy as np
from path import DATA_PATH
from analyze_data import npy_vertices_files, npy_faces_files, npy_groups_files
import matplotlib.pyplot as plt

#normalize the points s.t. they fit in a cube, mean = 0 stddev = 1; values are in range [-1,1]
def normalize_to_cube(points:np.array) -> np.array:
    # Ensure the points are a numpy array
    points = np.asarray(points)
    
    # Find the min and max along each axis (x, y, z)
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    
    # Compute the size of the bounding box along each axis
    box_size = max_vals - min_vals
    
    # Compute the center of the bounding box
    center = (min_vals + max_vals) / 2
    
    # Translate the object to the origin (move the center to [0, 0, 0])
    points_translated = points - center
    
    # Scale the object to fit inside a unit cube
    max_dim = np.max(box_size)
    points_normalized = points_translated / max_dim
    
    return points_normalized

#scale defines boundaries for cube visualization e.g. [-1,1] creates a 3x3x3 cube with each axis going from -1 to 1
def visualize_normalized_image(normalized_image:np.array, scale:list) -> None: 
    # Visualize the normalized points in the 3D unit cube
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the normalized points
    ax.scatter(normalized_image[:, 0], normalized_image[:, 1], normalized_image[:, 2], color='b', s=10)

    # Set the labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the aspect ratio to be equal, so the cube proportions are correct
    ax.set_box_aspect([1, 1, 1])

    # Set the limits for the axes to be between 0 and 1 (for unit cube)
    ax.set_xlim(scale)
    ax.set_ylim(scale)
    ax.set_zlim(scale)

    # Display the plot
    plt.show()

if __name__ == "__main__":
    # Example usage
    image = np.load(DATA_PATH + npy_vertices_files[0])
    faces = np.load(DATA_PATH + npy_faces_files[0])
    segm = np.load(DATA_PATH + npy_groups_files[0])
    print("NORMALIZATION INTO CUBE")
    normalized_image = normalize_to_cube(image)
    print(image)
    print(normalized_image)
    print("min value after normalizing")
    print(np.min(normalized_image, axis=0)) 
    print("max value after normalizing")
    print(np.max(normalized_image, axis=0)) 
    print("END OF NORMALIZATION")
    visualize_normalized_image(normalized_image,[-1,1])
    #plotMesh(image, faces, segm, figure_title='Mesh plot', fig='', save_figure_path='', row='', col='', scene_name='', show_grid=True, hover_title='', opacity=1)