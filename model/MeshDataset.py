import os
import numpy as np
import torch
from torch.utils.data import Dataset
from model.FaceGraphUNetModel import mesh_to_vertex_graph
from path import DATA_PATH
from preprocessing import normalize_to_cube
from model.MeshObject import MeshObject

class CustomMeshDataset(Dataset):
    def __init__(self, data_dir = DATA_PATH):
        """
        Custom Dataset for loading mesh data (vertices, faces, labels) stored in numpy files.
        
        Args:
            data_dir (str): Directory containing the mesh numpy arrays.
        """
        self.data_dir = data_dir
        self.mesh_files = [f for f in os.listdir(data_dir) if f.endswith('vertices.npy')]  # Assuming files are stored as .npy
        self.meshes = self._load_meshes()
    
    def _load_meshes(self):
        """
        Loads mesh data (vertices, faces, and labels) from .npy files stored in the directory.
        
        Returns:
            List of mesh data for each object.
        """
        meshes = []
        for vertex_file in self.mesh_files:
            if 'vertices' in vertex_file: 
                base_name = vertex_file.replace('vertices.npy', '')  # Assuming the naming convention is consistent
                
                # Load the corresponding files for each mesh
                vertices_path = os.path.join(self.data_dir, f"{base_name}vertices.npy")
                faces_path = os.path.join(self.data_dir, f"{base_name}faces.npy")
                labels_path = os.path.join(self.data_dir, f"{base_name}groups.npy")
            
                # Load the numpy arrays
                vertices = np.load(vertices_path)
                faces = np.load(faces_path)
                labels = np.load(labels_path)
                
                # Normalize the vertices
                vertices = normalize_to_cube(vertices)
                
                # Convert to PyTorch tensors
                vertices = torch.tensor(vertices, dtype=torch.float32)  # Shape: (N, 3)
                faces = torch.tensor(faces, dtype=torch.int64)  # Shape: (M, 3) or (M, K)
                labels = torch.tensor(labels, dtype=torch.int64)  # Shape: (M,)
                
                # Store the mesh data
                mesh_object = MeshObject(vertices=vertices, faces=faces, labels=labels)
                meshes.append(mesh_object)
            return meshes
    
    def __len__(self):
        return len(self.meshes)
    
    def __getitem__(self, idx):
        # Get a mesh object
        mesh = self.meshes[idx]
        
        # Extract the graph data
        graph_data = mesh_to_vertex_graph(mesh)

        # Return the graph data to the model
        return graph_data