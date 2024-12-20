from model.FaceGraphUNetModel import FaceGraphUNetModel
import torch
from torch.utils.data import DataLoader
from functions.plotAF_plotMesh_Juan import plotMesh_Juan
from model.MeshDataset import CustomMeshDataset


if __name__ == '__main__':
    # Instantiate device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a dataset: 
    # Load the data from the directory, split into train and test set in future work
    dataset = CustomMeshDataset()
    
    # Instantiate the DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Instantiate parameters (use same as in training)
    in_channels = 3
    hidden_channels = 32
    out_channels = 12
    depth = 3
    
    # Load the saved model
    model = FaceGraphUNetModel(in_channels,hidden_channels,out_channels,depth)
    model.load_state_dict(torch.load('trained_model.pth', map_location=device))
    model.eval()

    # Fetch a single batch from the dataloader (note: we do not separate into a train and test dataset here, this can be done in future work)
    single_batch = next(iter(dataloader))
    # Vertices
    inputs_x = single_batch['x'].to(device)
    # Edges
    inputs_edges = single_batch['edge_index'].to(device)
    # Groups
    inputs_y = single_batch['y'].to(device)
    # Faces 
    faces = single_batch['faces'].to(device)

    # Squeeze due to added batch dimension
    if inputs_edges.ndim == 3: 
        inputs_x = inputs_x.squeeze(0)
        inputs_edges = inputs_edges.squeeze(0)
        inputs_y = inputs_y.squeeze(0)
        faces = faces.squeeze(0)

    # Make the Prediction
    with torch.no_grad():
        # Feed input data into trained model
        output = model(inputs_x, inputs_edges)
        # Predict group for each vertex
        _, pred_labels = torch.max(output, 1)  # Output group labels for input data
    
    # Convert torch tensors to numpy arrays for visualization
    vertices_numpy = inputs_x.cpu().numpy()
    pred_numpy = pred_labels.cpu().numpy()
    faces = faces.cpu().numpy()
    y_labels = inputs_y.cpu().numpy()
    
    # Visualize predicted classes, note that data here is still normalized (since normalization is part of the dataloader)
    plotMesh_Juan(
            vertices_numpy, faces, pred_numpy,
            clim_limits=[0, 11], intensity_mode='vertex',
            color_scheme='Turbo',
            representation='groups_points',
            save_images=True, save_figure_path="out/sampleprediction.html"
    )

    # Visualize normalized original
    plotMesh_Juan(
            vertices_numpy, faces, y_labels,
            clim_limits=[0, 11], intensity_mode='vertex',
            color_scheme='Turbo',
            representation='groups_points',
            save_images=True, save_figure_path="out/originaldata.html"
    )