import os
from path import DATA_PATH
import numpy as np

# Import numpy files for faces, groups, and vertices
npy_faces_files = sorted([file for file in os.listdir(DATA_PATH) if file.endswith("faces.npy")])
npy_groups_files = sorted([file for file in os.listdir(DATA_PATH) if file.endswith("groups.npy")])
npy_vertices_files = sorted([file for file in os.listdir(DATA_PATH) if file.endswith("vertices.npy")])

if __name__ == "__main__":
    # Definition of areas for Faces
    print("----Faces-----")
    # Select a sample vector in data
    npy_faces_file_path = DATA_PATH / npy_faces_files[0]  # Construct the full path using pathlib
    somevector = np.load(npy_faces_file_path)
    # Print shape of vector
    print(somevector.shape)
    # Print example vector
    print(somevector)
    print("----END-----")

    # Definition of areas for Vertices
    print("----Vertices-----")
    # Select a sample vector in data
    npy_vertices_file_path = DATA_PATH / npy_vertices_files[0]  # Construct the full path using pathlib
    somevector = np.load(npy_vertices_file_path)
    # Print shape of vector
    print(somevector.shape)
    # Print example vector
    print(somevector)
    print("----END-----")

    # Definition of areas for Groups
    print("----Groups-----")
    # Select a sample vector in data
    npy_groups_file_path = DATA_PATH / npy_groups_files[0]  # Construct the full path using pathlib
    somevector = np.load(npy_groups_file_path)
    # Print shape of vector
    print(somevector.shape)
    # Print example vector
    print(somevector)
    print("----END-----")
