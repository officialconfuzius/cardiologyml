# Set the working directory correctly
# 1. File-> Settings
#
# 2. Build, Execution, Deployment -> Console -> Python Console
#
# 3. Working directory: [The path to the directory where the file you're currently working on resides.]
import numpy as np
import glob
import os

#%% HYBRID-ANALYSIS
VERBOSE = True
SHOW_FIGURES = True
OVERWRITE = False

SAVE_FIGURES = True
SAVE_FILES = True
#%%
import warnings
warnings.filterwarnings('ignore')

#%% PATHS
segmentation_database_path = 'C:/Users/msans/Downloads/UC3M/Machine Learning/SegmentationsPablo/'

# Ensure the output directory exists
output_directory = 'C:/Users/msans/Downloads/UC3M/Code/output/'  # Use your preferred output directory

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

#%% IMPORTS
from functions.importOBJ_functions import *

#%% GET SEGMENTATION FILES TO REVIEW
segmentation_files = sorted(glob.glob(segmentation_database_path + '*.obj'))

for s, segmentation_file_path in enumerate(segmentation_files):
    
    segmentation_file = segmentation_file_path.split('\\')[-1]
    aux_print = '  - Segmentation for ' + segmentation_file
    print(aux_print)

    # Get manual segmentation (if available)
    Segmentation_vertices, Segmentation_faces, Segmentation_groups = importOBJ(segmentation_file_path, variables_path=segmentation_file_path, SAVE_FILES=True, OVERWRITE_MESH_FILES=False)

    # PLOT FIGURE
    aux_figure_title = segmentation_file + ' - Segmentation'
    aux_save_figure_file = aux_figure_title + '.html'

    # Construct the absolute path for saving the figure
    save_figure_path = os.path.join(output_directory, aux_save_figure_file)

    plotMesh_Juan(Segmentation_vertices, Segmentation_faces, Segmentation_groups, clim_limits=[0, 11], intensity_mode='vertex',
                color_scheme='Turbo', figure_title=aux_figure_title,
                representation='groups_points',
                save_images=True, save_figure_path=save_figure_path)
    