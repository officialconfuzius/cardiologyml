import torch

class MeshObject:
    def __init__(self, vertices, faces, labels):
        """
        Initialize the mesh object using PyTorch tensors.
        
        Args:
            vertices (torch.Tensor): Vertex coordinates, shape (N, 3).
            faces (torch.Tensor): Face vertex indices, shape (M, K).
            labels (torch.Tensor): Labels for each vertex, shape (M,).
        """
        self.vertices = vertices.clone().detach()  # If vertices is already a tensor
        self.faces = faces.clone().detach()
        self.labels = labels.clone().detach()
        
        # Validation
        self._validate()

    def _validate(self):
        """
        Validates the integrity of the mesh representation.
        """
        num_vertices = self.vertices.shape[0]
        if (self.faces >= num_vertices).any() or (self.faces < 0).any():
            raise ValueError(f"Face indices must be in range [0, {num_vertices - 1}].")
        
        if self.vertices.shape[0] != self.labels.shape[0]:
            raise ValueError("Number of vertices must match the number of face labels.")
    
    def get_face_centroids(self):
        """
        Compute the centroids of each face.
        """
        centroids = torch.mean(self.vertices[self.faces], dim=1)  # Average of vertex positions
        return centroids
    
    def __repr__(self):
        return (f"MeshObject(\n"
                f"  Vertices: {self.vertices.shape[0]} points,\n"
                f"  Faces: {self.faces.shape[0]} faces,\n"
                f"  Face Labels: {self.labels.shape[0]} labels\n)")
    
    def getVertices(self): 
        return self.vertices
    
    def getFaces(self): 
        return self.faces
    
    def getLabels(self):
        return self.labels

# Example Usage
if __name__ == "__main__":
    # Sample data
    vertices = [
        [0.0, 0.0, 0.0],  # Vertex 0
        [1.0, 0.0, 0.0],  # Vertex 1
        [0.0, 1.0, 0.0],  # Vertex 2
        [0.0, 0.0, 1.0]   # Vertex 3
    ]
    faces = [
        [0, 1, 2],  # Face 0
        [0, 2, 3],  # Face 1
        [0, 3, 1],  # Face 2
        [1, 3, 2]   # Face 3
    ]
    labels = [1, 2, 1, 3]  # Labels for each face

    # Create the MeshObject
    mesh = MeshObject(vertices, faces, labels)

    # Display the MeshObject
    print(mesh)

    # Compute and display centroids
    centroids = mesh.get_face_centroids()
    print("Face centroids:", centroids)