import torch
from torch_geometric.nn import GraphUNet
from torch_geometric.data import Data
from model.MeshObject import MeshObject

class FaceGraphUNetModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth):
        super(FaceGraphUNetModel, self).__init__()
        
        # Define the Graph UNet model
        self.unet = GraphUNet(in_channels, hidden_channels, out_channels, depth)

    def forward(self, x, edge_index):
        # x: Node features (e.g., vertex features)
        # edge_index: Graph connectivity (e.g., faces that connect vertices)
        
        # Apply Graph UNet to predict labels per vertex
        return self.unet(x, edge_index)



def mesh_to_vertex_graph(mesh):
    """
    Convert a MeshObject to a graph where nodes represent vertices.
    
    Args:
        mesh (MeshObject): The input MeshObject.
    
    Returns:
        torch_geometric.data.Data: Graph representation where nodes are vertices.
    """
    # Node features: Use vertex coordinates as features
    vertex_features = mesh.vertices  # Shape (N, 3), where N is the number of vertices

    # Create edges based on vertex adjacency (vertices sharing an edge in any face)
    edges = []
    for face in mesh.faces:
        num_vertices_in_face = len(face)
        for i in range(num_vertices_in_face):
            for j in range(i + 1, num_vertices_in_face):
                # Add edge between vertex i and vertex j in this face
                edges.append((face[i], face[j]))
                edges.append((face[j], face[i]))  # Ensure bidirectionality

    # Convert edges list into tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Ensure that edges are bidirectional and remove duplicates
    edge_index = torch.unique(edge_index, dim=1)

    # Labels: Use vertex labels directly
    labels = mesh.labels  # Shape (N,), where N is the number of vertices
    
    # Return Data object
    return {
        "x":vertex_features.clone().detach(),  # Node features (vertex coordinates)
        "edge_index":edge_index,               # Edge indices (connectivity between vertices)
        "y":labels.clone().detach()            # Vertex labels (for segmentation)
    }

if __name__ == '__main__':
     # Create a MeshObject
    vertices = torch.tensor([
        [0.0, 0.0, 0.0],  # Vertex 0
        [1.0, 0.0, 0.0],  # Vertex 1
        [0.0, 1.0, 0.0],  # Vertex 2
        [0.0, 0.0, 1.0]   # Vertex 3
    ], dtype=torch.float32)

    faces = torch.tensor([
        [0, 1, 2],  # Face 0
        [0, 2, 3],  # Face 1
        [0, 3, 1],  # Face 2
        [1, 3, 2]   # Face 3
    ], dtype=torch.int64)

    face_labels = torch.tensor([1, 2, 1, 3], dtype=torch.int64)  # Labels for each face

    # Create a MeshObject that encapsulates all the crucial information about the object
    mesh = MeshObject(vertices, faces, face_labels)
    # Example Usage
    graph_data = mesh_to_vertex_graph(mesh)