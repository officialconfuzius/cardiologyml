import os
import numpy as np
DATA_PATH = 'C:/Users/msans/Downloads/UC3M/Machine Learning/SegmentationsPablo/'


#import numpy files for faces, groups and vertices
npy_faces_files = sorted([file for file in os.listdir(DATA_PATH) if file.endswith("faces.npy")])
npy_groups_files = sorted([file for file in os.listdir(DATA_PATH) if file.endswith("groups.npy")])
npy_vertices_files = sorted([file for file in os.listdir(DATA_PATH) if file.endswith("vertices.npy")])


if __name__ == "__main__":
    #defintion of areas?
    print("----Faces-----")
    #select a sample vector in data
    somevector = np.load(DATA_PATH + npy_faces_files[0])
    #print shape of vector
    print(somevector.shape)
    #print example vector
    print(somevector)
    print("----END-----")

    #definition of end of areas?
    print("----Vertices-----")
    #select a sample vector in data
    somevector = np.load(DATA_PATH + npy_vertices_files[0])
    #print shape of vector
    print(somevector.shape)
    #print example vector
    print(somevector)
    print("----END-----")

    #definition of a manually assigned group?
    print("----Groups-----")
    #select a sample vector in data
    somevector = np.load(DATA_PATH + npy_groups_files[0])
    #print shape of vector
    print(somevector.shape)
    #print example vector
    print(somevector)
    print("----END-----")