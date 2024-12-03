import os
import sys
import time
import numpy as np
import scipy.stats as scst
from scipy.spatial import distance
import functions.objloader as objl
from functions.meshFusion import *
from functions.plotAF_plotMesh_Juan import plotMesh_Juan

def fix_orphan_points(groups, face_loc):
    """
    Uses the KNN method to assign the orphan points to the most prominent class in the vicinity

    Parameters
    ----------
    groups
    faces

    Returns
    -------
    faces_fixed
    """
    # GROUP CORRECTION BY KNN
    point = face_loc[groups == 0] #faces without group
    Search_points = face_loc[groups != 0] #faces locations
    segmentation_radius = 50 #5cm
    num_return_indices = 5
    [closest_face_indices, closest_face_distances] = findClosestPointsByDistance(point, Search_points, segmentation_radius, num_return_indices)
    closest_face_groups_array = groups[closest_face_indices-1]
    if point.shape[0] == 1:
        closest_face_groups = scst.mode(closest_face_groups_array).mode.squeeze()
    else:
        closest_face_groups = scst.mode(closest_face_groups_array, axis=1).mode.squeeze()
    groups_fixed = groups.copy()
    groups_fixed[groups_fixed == 0] = closest_face_groups
    groups_fixed -= 1
    return groups_fixed

def vertex_groups(groups, vertex, face_loc):
    """
    Assign to each vertex the group of the nearest face
    Parameters
    ----------
    groups
    vertex
    face_loc

    Returns
    -------
    v_groups
    """
    v_groups = np.array([])
    aux_steps = np.arange(0, vertex.shape[0], 5000)
    aux_steps = np.unique(np.append(aux_steps, vertex.shape[0])) #unique in case the last position was already included in the range
    for ii, step in enumerate(aux_steps):
        print('------ ' + str(step) + '/' + str(vertex.shape[0]) + ' vertices assigned')
        if ii < (aux_steps.shape[0]-1):
            point = vertex[step:aux_steps[ii+1], :] # don't have group assigned
            Search_points = face_loc # each face has an assigned group
            segmentation_radius = 5 #5mm
            num_return_indices = 1
            [closest_face_indices, closest_face_distances] = findClosestPointsByDistance(point, Search_points, segmentation_radius, num_return_indices)
            aux_group = groups[closest_face_indices]
            v_groups = np.append(v_groups, aux_group)
    return v_groups

def importOBJ(segmentation_file_path, variables_path, SAVE_FILES = True, OVERWRITE_MESH_FILES = False):
    start_time = time.time()
    # Check .obj file
    print('---> Looking for OBJ file')
    file_exists = os.path.exists(segmentation_file_path)
    if file_exists:
        print('------> OBJ file found')
        print('------> Looking for variables files')
        var_exists = [os.path.exists(variables_path + '.vertices.npy'), os.path.exists(variables_path + '.faces.npy'), os.path.exists(variables_path + '.groups.npy')]
        PROCESS_OBJ = True
        if (sum(var_exists) == 3):# & not(OVERWRITE_MESH_FILES):
            print('--------> Variables files found')
            aux_vertices = np.load(variables_path + '.vertices.npy')
            aux_faces = np.load(variables_path + '.faces.npy')
            groups_vertex = np.load(variables_path + '.groups.npy')
        else:
            print('--------x Variables files incomplete -> generating files...')
            print('---> Loading OBJ file')

            aux_obj = objl.ObjFile(segmentation_file_path)

            # Load .obj
            aux_vertices_list = aux_obj.vertices
            aux_faces_list = aux_obj.faces

            num_vertices = len(aux_vertices_list)
            num_faces = len(aux_faces_list)
            print('-------- num_vertices: ', num_vertices)
            print('-------- num_faces: ', num_faces)

            # List --> numpy
            aux_vertices = np.array(aux_vertices_list)
            # print(aux_vertices.shape)

            aux_faces = [] #np.array((num_faces, 3))
            aux_segmentation = []
            aux_face_loc = np.empty((1, 3))
            quantiles = np.floor(np.quantile(np.arange(num_faces),[0.25,0.50,0.75,1]))
            for f, aux_face in enumerate(aux_faces_list):
                if f in quantiles:
                    aux_quant = (np.where(f==quantiles)[0][0] + 1)*25
                    print('---------- ' + str(aux_quant) + '% of the faces')
                aux_s = aux_face[3]
                aux_f = aux_face[0]
                aux_l = np.mean(aux_vertices[np.array(aux_f)-1], axis=0)
                aux_faces.append(aux_f)
                aux_segmentation.append(aux_s)
                aux_face_loc = np.vstack((aux_face_loc, aux_l))
            aux_faces = np.array(aux_faces) - 1  # faces indexing starts at 1, not 0
            aux_segmentation = np.array(aux_segmentation)
            aux_face_loc = aux_face_loc[1:, :]

            # Face groups preserving order
            _, idx = np.unique(aux_segmentation, return_index=True)
            segmentation_group_names = aux_segmentation[np.sort(idx)]
            num_segmentation_groups = len(segmentation_group_names)
            print('-------- num_segmentation_groups: ', num_segmentation_groups)
            group_dict = dict()
            for gg, group in enumerate(segmentation_group_names):
                group_dict[group] = gg

            aux_segmentation_ordered = np.zeros(aux_segmentation.shape[0])
            for gg, g_str in enumerate(aux_segmentation):
                aux_segmentation_ordered[gg] = group_dict[g_str] #int(g_str[:].replace('mmGroup',''))

            # ORPHAN POINTS
            print('---> Looking for orphan points')
            if len(segmentation_group_names) != 12:
                print('------> Orphan points found -> assigning points with KNN')
                groups_fixed = fix_orphan_points(aux_segmentation_ordered, aux_face_loc)
                orphan = 1
            else:
                print('------> No orphan points found')
                groups_fixed = aux_segmentation_ordered
                orphan = 0
            print('---> Assigning groups to vertices')
            groups_vertex = vertex_groups(groups_fixed, aux_vertices, aux_face_loc)


            filename_vertex = variables_path + '.vertices.npy'
            filename_face = variables_path + '.faces.npy'
            filename_groups = variables_path + '.groups.npy'
            if SAVE_FILES:
                print('---> Storing vertex, face and group variables')
                with open(filename_vertex, 'wb') as f:
                    np.save(f, aux_vertices)
                with open(filename_face, 'wb') as f:
                    np.save(f, aux_faces)
                with open(filename_groups, 'wb') as f:
                    np.save(f, groups_vertex)

    else:
        print('------x OBJ file not found: ' + segmentation_file_path)
        aux_vertices = []
        aux_faces = []
        groups_vertex = []
    return aux_vertices, aux_faces, groups_vertex
