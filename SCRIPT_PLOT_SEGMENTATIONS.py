# Set the working directory correctly
# 1. File-> Settings
# 2. Build, Execution, Deployment -> Console -> Python Console
# 3. Working directory: [The path to the directory where the file you're currently working on resides.]

import numpy as np
import os
from pathlib import Path
import glob
import warnings
from path import DATA_PATH  # Importing DATA_PATH from path.py

#%% HYBRID-ANALYSIS
VERBOSE = True
SHOW_FIGURES = True
OVERWRITE = False

SAVE_FIGURES = True
SAVE_FILES = True

warnings.filterwarnings('ignore')

#%% PATHS
segmentation_database_path = DATA_PATH  # Using DATA_PATH imported from path.py
output_directory = DATA_PATH.parent / "output"  # Define the output folder relative to the segmentation path

# Ensure the output directory exists
output_directory.mkdir(parents=True, exist_ok=True)

# Debugging: Verify if the segmentation database path exists and list files in it
if not segmentation_database_path.exists():
    print(f"Segmentation path does not exist: {segmentation_database_path}")
else:
    print(f"Segmentation path exists: {segmentation_database_path}")
    # List all files in this path for debugging
    for file in segmentation_database_path.iterdir():
        print(f"File in segmentation path: {file}")

#%% GET SEGMENTATION FILES TO REVIEW
# Using pathlib to find .obj files directly
segmentation_files = sorted(glob.glob(str(segmentation_database_path / '*.obj')))

# Debugging: Print the number of files found
print(f"Found {len(segmentation_files)} segmentation files.")
for file in segmentation_files:
    print(f"Segmentation file: {file}")

#%% IMPORTS
from functions.importOBJ_functions import *
# atrial_areas = ['LAA', 'LSPV', 'LIPV', 'RIPV', 'RSPV', 'PW', 'AR', 'AW', 'AFL', 'AS', 'LW', 'MV']
# (Optional information about atrial areas, which might be useful for later analysis)

#%% PROCESS EACH SEGMENTATION FILE
# Loop over all segmentation files and process them one by one
for s, segmentation_file_path in enumerate(segmentation_files):
    segmentation_file = os.path.basename(segmentation_file_path)  # Extract the filename from the full path
    aux_print = '  - Segmentation for ' + segmentation_file
    print(aux_print)

    try:
        # Get manual segmentation (if available)
        print("Calling importOBJ...")
        Segmentation_vertices, Segmentation_faces, Segmentation_groups = importOBJ(
            segmentation_file_path,  # Use the string path as expected by importOBJ
            variables_path=segmentation_file_path,
            SAVE_FILES=True,
            OVERWRITE_MESH_FILES=False
        )
        print("importOBJ completed successfully.")

        # PLOT FIGURE
        aux_figure_title = segmentation_file + ' - Segmentation'  # Title for the figure
        aux_save_figure_file = aux_figure_title + '.html'  # Output filename for the plot

        # Construct the absolute path for saving the figure
        save_figure_path = output_directory / aux_save_figure_file
        print(f"Plotting and saving figure to: {save_figure_path}")

        # Plot the segmentation and save it
        plotMesh_Juan(
            Segmentation_vertices, Segmentation_faces, Segmentation_groups,
            clim_limits=[0, 11], intensity_mode='vertex',
            color_scheme='Turbo', figure_title=aux_figure_title,
            representation='groups_points',
            save_images=True, save_figure_path=str(save_figure_path)
        )
        print("Plot saved successfully.")

    except Exception as e:
        # Handle exceptions so that the processing continues for other files
        print(f"Error processing {segmentation_file}: {e}")
        continue
