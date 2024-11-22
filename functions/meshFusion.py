"""Help on built-in module meshFusion:

NAME
    meshFusion

DESCRIPTION
    This module provides the necessary functions to merge different 3D atrial meshes to a target mesh

CLASSES
    ...
    ...

FUNCTIONS
    ...(...)
        Calculates ...
        Returns None

"""
import numpy as np
import sklearn.neighbors as skn
import vtk
from vtk.util import numpy_support
from scipy.spatial import distance
import numpy.matlib
from functions.plotAF import *
from scipy.interpolate import *
from tqdm import tqdm

# FUNCTION nonrigidICPv1()
def nonrigidICPv1(targetV, sourceV, targetF, sourceF, iterations, apply_alignment, verbose):
    """
    Function: nonrigidICPv1(targetV, sourceV, targetF, sourceF, iterations, apply_alignment, verbose)

    Parameters:
        targetV (numpy float array) : vertices of target mesh. Size [NumVertex_target, 3]
        sourceV (numpy float array) :  vertices of source mesh. Size [NumVertex_source, 3]
        targetF (numpy int array) : faces (triangles) of the target mesh. Size [NumTriangle_source, 3]
        targetV (numpy int array) : faces (triangles) of the source mesh. Size [NumTriangle_source, 3]
        iterations (int) : number of iterations. Usually 10-30 iterations
        apply_alignment (bool) : If True align the meshes. If False apply deformation directly
        verbose (bool): Print additional information

    Raises:
        ...

    Returns:
        registered (numpy float array) : registered source vertices on target mesh. Size [NumVertex_source, 3]
    """
    registered = []

    # Remove duplicate vertices
    targetV, indices_b = np.unique(targetV, axis=0, return_inverse=True)
    targetF = indices_b[targetF]

    # Assesment of meshes quality and simplification/improvement
    if verbose:
        aux_print = 'Remeshing and simplification target Mesh'
        print(aux_print)

    [cutoff, stdevui] = definecutoff(sourceV, sourceF)

    Indices_edgesS = detectedges(sourceV, sourceF)
    Indices_edgesT = detectedges(targetV, targetF)

    # Check free edges
    if len(Indices_edgesS) > 0:
        aux_print = 'Warning: Source mesh presents free edges. '
        print(aux_print)
        if apply_alignment:
            aux_print = 'Source Mesh presents free edges. Preallignement can not reliably be executed'
            print(aux_print)

    if len(Indices_edgesT) > 0:
        aux_print = 'Warning: Target mesh presents free edges. '
        print(aux_print)
        if apply_alignment:
            aux_print = 'Target Mesh presents free edges. Preallignement can not reliably be executed'
            print(aux_print)

    # Initial alignment and scaling
    if verbose:
        aux_print = 'Applying Rigid alignment'
        print(aux_print)
        print(aux_print)

    if apply_alignment:
        # TO BE DONE
        aux_print = ' TO BE DONE. Alignment'
        print(aux_print)
        [error1, sourceV, transform] = rigidICP(targetV, sourceV, Indices_edgesS, Indices_edgesT)

    p = sourceV.shape[0]

    # General deformation
    if verbose:
        aux_print = 'General deformation'
        print(aux_print)
    kernel1_step = 0.5/iterations
    kernel2_step = 0.3/iterations
    kernel1 = np.arange(1.5, 1 - kernel1_step, -kernel1_step)
    kernel2 = np.arange(2.4, 2.7, kernel2_step)

    # HASTA AQUI 100% COMPROBADO (sin alignment)
    for it in range(iterations):

        aux_print = 'it: ' + str(it) + '/' + str(iterations)
        print(aux_print)
        nrseedingpoints = np.round(np.power(10, kernel2[it]))

        [aux_idx1, aux_dist1] = meshknnsearch(targetV, sourceV)
        [aux_idx2, aux_dist2] = meshknnsearch(sourceV, targetV)
        aux_linear_idx1 = np.arange(0, sourceV.shape[0], 1)
        aux_linear_idx2 = np.arange(0, targetV.shape[0], 1)
        aux_linear_idx1 = np.reshape(aux_linear_idx1, aux_idx1.shape)
        aux_linear_idx2 = np.reshape(aux_linear_idx2, aux_idx2.shape)

        IDX1 = np.hstack((aux_idx1, aux_dist1, aux_linear_idx1))
        IDX2 = np.hstack((aux_idx2, aux_dist2, aux_linear_idx2))

        # C = SETDIFF(A,B) for vectors A and B, returns the values in A that
        # are not in B with no repetitions. C will be sorted
        # If there are repeated values in A that are not in B, then the index of the first
        # occurrence of each repeated value is returned.
        # TARGET
        C1 = np.setdiff1d(aux_idx1, Indices_edgesT)
        ia = np.empty((0, 0))
        for j, aux_C in enumerate(C1):
            aux_find = np.where(aux_idx1 == aux_C)[0]
            if len(aux_find) > 0:
                # Append 1st occurrence
                ia = np.append(ia, aux_find[0])
        ia = ia.astype(int)
        IDX1 = IDX1[ia, :]

        # SOURCE
        C2 = np.setdiff1d(aux_idx2, Indices_edgesS)
        ia = np.empty((0, 0))
        for j, aux_C in enumerate(C2):
            aux_find = np.where(aux_idx2 == aux_C)[0]
            if len(aux_find) > 0:
                # Append 1st occurrence
                ia = np.append(ia, aux_find[0])
        ia = ia.astype(int)
        IDX2 = IDX2[ia, :]


        sourcepartial = sourceV[IDX1[:, 2].astype(int), :]
        targetpartial = targetV[IDX2[:, 2].astype(int), :]

        [IDXS, dS] = meshknnsearch(targetpartial, sourcepartial)
        [IDXT, dT] = meshknnsearch(sourcepartial, targetpartial)

        # HASTA AQUI OK
        ppartial = sourcepartial.shape[0]
        reduction = nrseedingpoints/sourceV.shape[0]

        # Esta funcion cambia (no habia alternativa python)
        if verbose:
            aux_print = ' - Apply reduction: ' + str(100*reduction) + '%'
            print(aux_print)
            aux_print = ' - Expected reduced faces size: ' + str(reduction*sourceF.shape[0])
            print(aux_print)

        decimatedPoly = reduceMeshPatch(sourceF, sourceV, reduction, False, False)

        vtk_points = decimatedPoly.GetPoints().GetData()
        tempV = numpy_support.vtk_to_numpy(vtk_points)
        num_decimated_faces = decimatedPoly.GetNumberOfPolys()
        vtk_faces = decimatedPoly.GetPolys().GetData()
        tempF = numpy_support.vtk_to_numpy(vtk_faces)
        tempF = np.reshape(tempF, (num_decimated_faces, 4))
        tempF = tempF[:, 1:]

        [idx, didx] = meshknnsearch(sourcepartial, tempV)
        q = idx.shape[0]

        D = distance.cdist(sourcepartial, tempV, 'euclidean')
        aux_D_mean = np.mean(np.mean(D))
        gamma = 1/(2*np.power(aux_D_mean, kernel1[it]))

        # HASTA AQUI TODO OK (Con adaptatición mesh decimation) # linea 123 en matlab nonrigidICPv1.m
        Datasetsource = np.vstack((sourcepartial, sourcepartial[IDXT,:].squeeze()))
        Datasettarget = np.vstack((targetpartial[IDXS,:].squeeze(), targetpartial))
        Datasetsource2 = np.vstack((D, D[IDXT].squeeze()))

        vectors = Datasettarget - Datasetsource
        r = vectors.shape[0]

        # Define radial basis width for deformation points gaussian
        tempy1 = np.exp(-gamma*np.power(Datasetsource2, 2))
        tempy2 = np.zeros((3*r, 3*q))
        tempy2[:r, :q] = tempy1
        tempy2[r:2*r, q:2*q] = tempy1
        tempy2[2*r:, 2*q:] = tempy1

        # Solve optimal deformation directions with regularization term
        # ppi=inv( (tempy2.'*tempy2)+lambda*eye(3*q)) *(tempy2.');
        aux_lambda = 0.001
        aux_matrix = np.matmul(tempy2.T, tempy2) + aux_lambda*np.identity(3*q)
        aux_matrix = np.linalg.inv(aux_matrix)
        ppi = np.matmul(aux_matrix, tempy2.T)

        aux_vectors = np.reshape(vectors, (3*r,1))
        modes = np.matmul(ppi, aux_vectors)

        D2 = distance.cdist(sourceV, tempV, 'euclidean')
        aux_D2_mean = np.mean(np.mean(D2))
        gamma2 = 1 / (2 * np.power(aux_D2_mean, kernel1[it]))

        tempyfull1 = np.exp(-gamma2*np.power(D2, 2))
        tempyfull2 = np.zeros((3*p, 3*q))
        tempyfull2[:p, :q] = tempyfull1
        tempyfull2[p:2*p, q:2*q] = tempyfull1
        tempyfull2[2*p:, 2*q:] = tempyfull1

        test2 = np.matmul(tempyfull2, modes)
        test2 = np.reshape(test2, (int(test2.shape[0]/3), 3))
        # deforme source mesh
        sourceV = sourceV+test2

        [error1, sourceV, transform] = rigidICP(targetV, sourceV, Indices_edgesS, Indices_edgesT)

    # Local deformation (line 174 nonrigidICPv1.m)
    aux_print = 'Local optimization'
    print(aux_print)

    arraymap = []
    kk = 12 + iterations
    normalsT = getVertexNormal(targetV, targetF) * cutoff
    normalsT[np.isnan(normalsT)] = 0

    # Define local mesh relation
    normalsS = getVertexNormal(sourceV, sourceF) * cutoff
    aux_local_mesh = np.column_stack((sourceV, normalsS))
    [IDXsource, Dsource] = meshknnsearch(aux_local_mesh, aux_local_mesh, k=kk)

    # check normal direction
    # REVISAR
    [IDXcheck, Dcheck] = meshknnsearch(targetV, sourceV, k=1)

    aux_substract = np.subtract(normalsS, normalsT[IDXcheck, :].squeeze())
    aux_add = np.add(normalsS, normalsT[IDXcheck, :].squeeze())
    testpos = np.sum(np.power(aux_substract, 2))
    testneg = np.sum(np.power(aux_add, 2))


    if testneg < testpos:
        normalsT = -normalsT
        aux_temp = targetF[:, 1]
        targetF[:, 1] = targetF[:, 2]
        targetF[:, 2] = aux_temp

    # for line 202 nonrigidICPv1
    for ddd in range(iterations):

        k = kk - iterations

        normalsS = getVertexNormal(sourceV, sourceF) * cutoff

        aux_Dsource = Dsource[:, :k]
        sumD = np.sum(aux_Dsource, axis=1)
        sumD2 = np.matmul(sumD.reshape(sumD.shape[0], 1), np.ones((1, k)))
        sumD3 = sumD2 - aux_Dsource
        sumD2 = sumD2 * (k-1)
        weights = sumD3/sumD2

        aux_search1 = np.column_stack((targetV, normalsT))
        aux_search2 = np.column_stack((sourceV, normalsS))
        [IDXtarget, Dtarget] = meshknnsearch(aux_search1, aux_search2, k=3)
        pp1 = targetV.shape[0]

        if len(Indices_edgesT) > 0:

            # np.sum(a == B) for a in A
            aux_A = IDXtarget[:, 0]
            aux_B = Indices_edgesT
            # aux_is_member0 = [np.sum(a == aux_B) for a in aux_A]
            aux_is_member0 = numpy.in1d(aux_A, aux_B)
            correctionfortargetholes0 = [i for i, x in enumerate(aux_is_member0) if x]
            aux_stack = sourceV[correctionfortargetholes0, :]
            targetV = np.vstack((targetV, aux_stack))
            IDXtarget[correctionfortargetholes0, 0] = pp1 + np.arange(len(correctionfortargetholes0))
            Dtarget[correctionfortargetholes0, 0] = 0.00001

            aux_A = IDXtarget[:, 1]
            aux_is_member1 = numpy.in1d(aux_A, aux_B)
            correctionfortargetholes1 = [i for i, x in enumerate(aux_is_member1) if x]
            pp = targetV.shape[0]
            aux_stack = sourceV[correctionfortargetholes1, :]
            targetV = np.vstack((targetV, aux_stack))
            IDXtarget[correctionfortargetholes1, 1] = pp + np.arange(len(correctionfortargetholes1))
            Dtarget[correctionfortargetholes1, 1] = 0.00001

            aux_A = IDXtarget[:, 2]
            aux_is_member2 = numpy.in1d(aux_A, aux_B)
            correctionfortargetholes2 = [i for i, x in enumerate(aux_is_member2) if x]
            pp = targetV.shape[0]
            aux_stack = sourceV[correctionfortargetholes2, :]
            targetV = np.vstack((targetV, aux_stack))
            IDXtarget[correctionfortargetholes2, 2] = pp + np.arange(len(correctionfortargetholes2))
            Dtarget[correctionfortargetholes2, 2] = 0.00001

        # line 242
        summD = np.sum(Dtarget, axis=1)
        summD2 = np.matmul(summD.reshape(summD.shape[0], 1), np.ones((1, 3)))
        summD3 = summD2 - Dtarget
        weightsm = summD3/(summD2*2)


        aux_tar0 = np.column_stack((weightsm[:, 0] * targetV[IDXtarget[:, 0], 0],
                                   weightsm[:, 0] * targetV[IDXtarget[:, 0], 1],
                                   weightsm[:, 0] * targetV[IDXtarget[:, 0], 2]))
        aux_tar1 = np.column_stack((weightsm[:, 1] * targetV[IDXtarget[:, 1], 0],
                                   weightsm[:, 1] * targetV[IDXtarget[:, 1], 1],
                                   weightsm[:, 1] * targetV[IDXtarget[:, 1], 2]))
        aux_tar2 = np.column_stack((weightsm[:, 2] * targetV[IDXtarget[:, 2], 0],
                                   weightsm[:, 2] * targetV[IDXtarget[:, 2], 1],
                                   weightsm[:, 2] * targetV[IDXtarget[:, 2], 2]))

        Targettempset = aux_tar0 + aux_tar1 + aux_tar2

        targetV = targetV[:pp1, :]

        arrayMap = []
        for i in range(sourceV.shape[0]):
            sourceset = sourceV[IDXsource[i, :k], :]
            targetset = Targettempset[IDXsource[i, :k], :]
            d, Z, tform = procrustes(targetset, sourceset, scaling=False)
            arrayMap.append(tform)

        # Line 258
        sourceVapprox = sourceV
        sourceVtemp = np.zeros((k, 3))
        for i in range(sourceV.shape[0]):
            for ggg in range(k):
                aux_p = np.reshape(sourceV[i, :], (1, 3))
                aux_sum = arrayMap[IDXsource[i, ggg]]['scale']*aux_p.dot(arrayMap[IDXsource[i, ggg]]['rotation'])
                aux_sum = aux_sum + arrayMap[IDXsource[i, ggg]]['translation']
                sourceVtemp[ggg, :] = weights[i, ggg] * aux_sum

            sourceV[i, :] = np.sum(sourceVtemp[:k, :], axis=0)

        sourceV = sourceVapprox + 0.5 * (sourceV - sourceVapprox)

    registered = sourceV
    extra = {'sourceV': sourceV, 'sourceF': sourceF, 'targetV': targetV, 'targetF': targetF}

    return registered, extra


# FUNCTION getVertexNormal()
def getVertexNormal(vertices, faces):
    # TO BE DONE
    # Each vertex belongs to one or more faces
    # Calculate the normal vertex for each triangle
    # Then for every vertex find all triangles including the vertex and average the norm value
    num_faces = faces.shape[0]
    norm_faces = np.zeros((num_faces, 3))
    for f in range(num_faces):
        # aux_print = 'f: ' + str(f)
        # print(aux_print)
        aux_f = faces[f, :]
        aux_p0 = vertices[aux_f, 0]
        aux_p1 = vertices[aux_f, 1]
        aux_p2 = vertices[aux_f, 2]

        # A - -- B
        #   \ /
        #    C
        aux_v1 = aux_p1 - aux_p0
        aux_v2 = aux_p2 - aux_p0
        aux_cp = np.cross(aux_v1, aux_v2)
        aux_cp_norm = np.linalg.norm(aux_cp)
        aux_f_norm = aux_cp/aux_cp_norm
        norm_faces[f, :] = aux_f_norm

    num_vertices = vertices.shape[0]
    norm_vertices = np.zeros((num_vertices, 3))
    for v in range(num_vertices):

        aux_indices0 = np.where(faces[:, 0] == v)[0]
        aux_indices1 = np.where(faces[:, 1] == v)[0]
        aux_indices2 = np.where(faces[:, 2] == v)[0]
        aux_indices = np.hstack((aux_indices0, aux_indices1, aux_indices2))

        aux_norm_faces = norm_faces[aux_indices, :]
        # print(aux_norm_faces)
        aux_norm_faces = np.sum(aux_norm_faces, axis=0)
        aux_norm_faces = aux_norm_faces/np.linalg.norm(aux_norm_faces)
        norm_vertices[v, :] = aux_norm_faces

    return norm_vertices





def rigidICP(targetV, sourceV, Indices_edgesS, Indices_edgesT):

    error1 = []
    transform = []

    Prealligned_source = sourceV
    Prealligned_target = targetV

    # [errortemp(index,:),Reallignedsourcetemp] = ICPmanu_allign2(Prealligned_target,Prealligned_source,Indices_edgesS,Indices_edgesT);
    [aux_error, Reallignedsourcetemp] = ICPmanu_allign2(Prealligned_target,Prealligned_source,Indices_edgesS,Indices_edgesT)

    error_stop_condition = 0.000001
    old_error = 1
    new_error = 100000
    error_array = np.empty((0, 0))
    error_diff = 100000

    while error_diff > error_stop_condition:

        old_error = new_error
        [new_error, Reallignedsourcetemp] = ICPmanu_allign2(Prealligned_target, Reallignedsourcetemp,
                                                                       Indices_edgesS, Indices_edgesT);
        error_array = np.append(error_array, new_error)
        error_diff = old_error - new_error
        print('Error value: ' + str(new_error))

        [d, Reallignedsource, transform] = procrustes(Reallignedsourcetemp, sourceV)

    return error_array, Reallignedsource, transform

def ICPmanu_allign2(target, source, Indices_edgesS, Indices_edgesT):

    [aux_idx1, aux_dist1] = meshknnsearch(target, source)
    [aux_idx2, aux_dist2] = meshknnsearch(source, target)
    aux_linear_idx1 = np.arange(0, source.shape[0], 1)
    aux_linear_idx2 = np.arange(0, target.shape[0], 1)
    aux_linear_idx1 = np.reshape(aux_linear_idx1, aux_idx1.shape)
    aux_linear_idx2 = np.reshape(aux_linear_idx2, aux_idx2.shape)

    IDX1 = np.hstack((aux_idx1, aux_dist1, aux_linear_idx1))
    IDX2 = np.hstack((aux_idx2, aux_dist2, aux_linear_idx2))

    C2 = np.setdiff1d(aux_idx2, Indices_edgesS)
    ia = np.empty((0, 0))
    for j, aux_C in enumerate(C2):
        aux_find = np.where(aux_idx2 == aux_C)[0]
        if len(aux_find) > 0:
            # Append 1st occurrence
            ia = np.append(ia, aux_find[0])
    ia = ia.astype(int)
    IDX2 = IDX2[ia, :]

    C1 = np.setdiff1d(aux_idx1, Indices_edgesT)
    ia = np.empty((0, 0))
    for j, aux_C in enumerate(C1):
        aux_find = np.where(aux_idx1 == aux_C)[0]
        if len(aux_find) > 0:
            # Append 1st occurrence
            ia = np.append(ia, aux_find[0])
    ia = ia.astype(int)
    IDX1 = IDX1[ia, :]

    m1 = np.mean(IDX1[:, 1])
    s1 = np.std(IDX1[:, 1])

    aux_dist_indices = np.where(IDX2[:, 1] < (m1 + 1.96*s1))[0]
    IDX2 = IDX2[aux_dist_indices, :]

    Datasetsource = np.vstack((source[IDX1[:,2].astype(int),:], source[IDX2[:,0].astype(int),:]))
    Datasettarget = np.vstack((target[IDX1[:, 0].astype(int), :], target[IDX2[:, 2].astype(int), :]))

    [error, Reallignedsource, transform] = procrustes(Datasettarget, Datasetsource)

    # tform = {'rotation': T, 'scale': b, 'translation': c}
    # Reallignedsource = transform.b * source * transform.T + repmat(transform.c(1, 1:3), size(source, 1), 1);
    aux_matrix = transform['scale'] * source
    aux_matrix = np.matmul(aux_matrix, transform['rotation'])
    aux_c = transform['translation']
    aux_c = np.reshape(aux_c, (1,3))
    aux_matrix2 = np.matlib.repmat(aux_c, source.shape[0], 1)

    Reallignedsource = aux_matrix + aux_matrix2

    return error, Reallignedsource

def reduceMeshPatch(faces, verts, reduction, SHOW_VTK=False, verbose=False):

    numFaces = faces.shape[0]
    # if reduction > 0 and reduction <= 1:
    #     reduction = numFaces * reduction
    # reduction = np.min([numFaces+1, reduction])

    # Specify the desired reduction in the total number of polygons
    # (e.g., if TargetReduction is set to 0.9, this filter will try to reduce
    # the data set to 10% of its original size).
    target_reduction = 1 - reduction # percentage of preserved points after reduction

    polydata = vtk.vtkPolyData()
    # vtk Points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(verts))
    polydata.SetPoints(vtk_points)
    # vtk cells/triangles
    # Triangles = vtk.vtkCellArray()
    # Triangle = vtk.vtkTriangle()
    #
    # for s in faces:
    #     Triangle.GetPointIds().SetId(0, s[0])
    #     Triangle.GetPointIds().SetId(1, s[1])
    #     Triangle.GetPointIds().SetId(2, s[2])
    #
    #     Triangles.InsertNextCell(Triangle)
    # polydata.SetPolys(Triangles)

    cells = vtk.vtkCellArray()
    # - Note that the cell array looks like this: [3 vtx0 vtx1 vtx2 3 vtx3 ... ]
    cells_npy = np.column_stack(
        [np.full(numFaces, 3, dtype=np.int64), faces.astype(np.int64)]
    ).ravel()
    cells.SetCells(numFaces, numpy_support.numpy_to_vtkIdTypeArray(cells_npy))
    polydata.SetPolys(cells)

    if SHOW_VTK:
        # Setup actor and mapper
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInputConnection(polydata.GetProducerPort())
        else:
            mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(10)

        # Setup render window, renderer, and interactor
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderer.AddActor(actor)

        renderWindow.Render()
        renderWindowInteractor.Start()

    print("Before decimation\n"
          "-----------------\n"
          "There are " + str(polydata.GetNumberOfPoints()) + " points.\n"
          "There are " + str(polydata.GetNumberOfPolys()) + " faces.\n")

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(target_reduction)
    decimate.Update()

    decimatedPoly = vtk.vtkPolyData()
    # decimatedPoly.ShallowCopy(decimate.GetOutput())
    decimatedPoly.DeepCopy(decimate.GetOutput())

    print("After decimation \n"
          "-----------------\n"
          "There are " + str(decimatedPoly.GetNumberOfPoints()) + " points.\n"
          "There are " + str(decimatedPoly.GetNumberOfPolys()) + " faces.\n")


    if SHOW_VTK:
        print('aaaaaaaaaa')
        # Setup actor and mapper
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInputConnection(decimatedPoly.GetProducerPort())
        else:
            mapper.SetInputData(decimatedPoly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(10)

        # Setup render window, renderer, and interactor
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderer.AddActor(actor)

        renderWindow.Render()
        renderWindowInteractor.Start()

    # # reducepatch() Matlab's code (adapted)
    # [frows, fcols] = faces.shape
    #
    # # SKIP (not implemented)
    # # [frows, fcols] = size(faces);
    # # if fcols>3   % triangulate using very simple algorithm
    # #   newfaces = zeros(frows*(fcols-2),3);
    # #   newfaces(:,1) = repmat(faces(:,1),fcols-2,1);
    # #   for k = 2:fcols-1
    # #     newfaces( (1:frows)+frows*(k-2), 2:3) = [faces(:,k) faces(:,k+1)];
    # #   end
    # #   faces = newfaces;
    # # end
    #
    # # Unique vertices
    # verts_u, inverse_indices = np.unique(verts, axis=0, return_inverse=True)
    # faces_u = inverse_indices[faces]
    #
    # numFaces = faces_u.shape[0]
    # if reduction > 0 and reduction <= 1:
    #     reduction = numFaces * reduction
    # reduction = np.min([numFaces+1, reduction])
    #
    # # SKIP
    # # %remove the nans
    # # if ~isempty(faces)
    # #   pos = isnan(faces(:,1));
    # #   faces(pos,:) = [];
    # #   pos = isnan(faces(:,2));
    # #   faces(pos,2) = faces(pos,1);
    # #   pos = isnan(faces(:,3));
    # #   faces(pos,3) = faces(pos,2);
    # # end
    return decimatedPoly



def meshknnsearch(Points_A, Points_B, k=1):

    tree = skn.KDTree(Points_A, leaf_size=4)
    dist, ind = tree.query(Points_B, k)

    return ind, dist

    # TO BE DONE
    # Points_A (numpy float array) : Points to be analyzed. Size [Na, 3]
    # Points_B (numpy float array) : Search points. Size [Nb, 3]
    # finds the nearest neighbor in Points_A for each point in Points_B
    # Na = Points_A.shape[0]
    # Nb = Points_B.shape[0]
    #
    # min_index = np.zeros((Nb,))
    # min_dist = np.zeros((Nb,))
    #
    # for i in range(Nb):
    #     aux_point = Points_B[i, :]
    #
    #     aux_diff = np.power(Points_A[:, 0] - aux_point[0], 2) + np.power(Points_A[:, 1] - aux_point[1], 2) + np.power(Points_A[: ,2] - aux_point[2], 2)
    #     aux_dist = np.sqrt(aux_diff)
    #     aux_index = np.argmin(aux_dist)
    #     min_index[i] = aux_index
    #     min_dist[i] = aux_dist[aux_index]

def detectedges(V, F):

    fk1 = F[:, 0]
    fk2 = F[:, 1]
    fk3 = F[:, 2]

    aux_fk_12 = np.vstack((fk1, fk2))
    aux_fk_13 = np.vstack((fk1, fk3))
    aux_fk_23 = np.vstack((fk2, fk3))

    ed1 = np.sort(aux_fk_12.T, axis=1)
    ed2 = np.sort(aux_fk_13.T, axis=1)
    ed3 = np.sort(aux_fk_23.T, axis=1)
    # ed1 = np.sort(aux_fk_12, axis=1)
    # ed2 = np.sort(aux_fk_13, axis=1)
    # ed3 = np.sort(aux_fk_23, axis=1)

    ed = np.vstack((ed1, ed2, ed3))
    # Unique values with same order
    aux_unique, unique_indices = np.unique(ed, axis=0, return_index=True)
    unique_indices_sorted = np.unique(unique_indices)
    esingle = ed[unique_indices_sorted, :]

    # dubbles
    edouble = np.delete(ed, unique_indices_sorted, axis=0)

    # Returns the data in A that is not in B, with no repetitions. C is in sorted order.
    # C = setdiff(A, B, 'rows') # Matlab
    # C = setdiff(esingle,edouble,'rows');
    esingle1_rows = esingle.view([('', esingle.dtype)] * esingle.shape[1])
    edouble_rows = edouble.view([('', edouble.dtype)] * edouble.shape[1])
    C = np.setdiff1d(esingle1_rows, edouble_rows).view(esingle.dtype).reshape(-1, edouble.shape[1])

    C_array = C.T
    Indices_edges = C_array.reshape(-1)

    return Indices_edges

def definecutoff(vold, fold):

    fk1 = fold[:, 0]
    fk2 = fold[:, 1]
    fk3 = fold[:, 2]

    num_vertices = vold.shape[0]
    num_triangles = fold.shape[0]

    aux_sum1 = np.sum(np.power((vold[fk1, :] - vold[fk2, :]), 2), axis=1)
    aux_sum2 = np.sum(np.power((vold[fk1, :] - vold[fk3, :]), 2), axis=1)
    aux_sum3 = np.sum(np.power((vold[fk2, :] - vold[fk3, :]), 2), axis=1)

    D1 = np.sqrt(aux_sum1)
    D2 = np.sqrt(aux_sum2)
    D3 = np.sqrt(aux_sum3)

    aver = np.mean( np.hstack((D1, D2, D3)))
    stdevui = np.std( np.hstack((D1, D2)))

    return aver, stdevui


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

""""
    function[aver, stdevui] = definecutoff(vold, fold)

    fk1 = fold(:, 1);
    fk2 = fold(:, 2);
    fk3 = fold(:, 3);

    numverts = size(vold, 1);
    numfaces = size(fold, 1);

    D1 = sqrt(sum((vold(fk1,:) - vold(fk2,:)).^ 2, 2));
    D2 = sqrt(sum((vold(fk1,:) - vold(fk3,:)).^ 2, 2));
    D3 = sqrt(sum((vold(fk2,:) - vold(fk3,:)).^ 2, 2));

    aver = mean([D1;
    D2;
    D3]);
    stdevui = std([D1;
    D2]);
"""


""""
function[registered, targetV, targetF, extra] = nonrigidICPv1(targetV, sourceV, targetF, sourceF, iterations,
                                                              flag_prealligndata)

% INPUT
% -target: vertices
of
target
mesh;
n * 3
array
of
xyz
coordinates
% -source: vertices
of
source
mesh;
n * 3
array
of
xyz
coordinates
% -Ft: faces
of
target
mesh;
n * 3
array
% -Fs: faces
of
source
mesh;
n * 3
array
% -iterations: number
of
iterations;
usually
between
10
en
30
% -flag_prealligndata: 0 or 1.
% 0 if the
data
still
need
to
be
roughly
alligned
% 1 if the
data is already
alligned(manual or landmark
based)

% OUTPUT
% -registered: registered
source
vertices
on
target
mesh.Faces
are
not affected and remain
the
same is before
the
registration(Fs).

% EXAMPLE

% EXAMPLE
1
demonstrates
full
allignement and registration
of
two
complete
% meshes
% load
EXAMPLE1.mat
% [registered] = nonrigidICPv1(targetV, sourceV, targetF, sourceF, 10, 0);

% EXAMPLE
2
demonstrates
registration
of
two
incomplete
meshes
% load
EXAMPLE2.mat
% [registered] = nonrigidICPv1(targetV, sourceV, targetF, sourceF, 10, 1);

if nargin ~= 6
error('Wrong number of input arguments')
end

tic
clf
% remove
duplicate
vertices

[targetV, indexm, indexn] = unique(targetV, 'rows');
targetF = indexn(targetF);

% assesment
of
meshes
quality and simplification / improvement
disp('Remeshing and simplification target Mesh');

[cutoff, stdevui] = definecutoff(sourceV, sourceF);

[Indices_edgesS] = detectedges(sourceV, sourceF);
[Indices_edgesT] = detectedges(targetV, targetF);

if isempty(Indices_edgesS) == 0
    disp('Warning: Source mesh presents free edges. ');
    if flag_prealligndata == 0
        error('Source Mesh presents free edges. Preallignement can not reliably be executed')
    end
end

if isempty(Indices_edgesT) == 0
    disp('Warning: Target mesh presents free edges. ');
    if flag_prealligndata == 0
        error('Target mesh presents free edges. Preallignement can not reliably be executed')
    end
end

% initial
allignment and scaling
disp('Rigid allignement source and target mesh');

if flag_prealligndata == 1
    [error1, sourceV, transform] = rigidICP(targetV, sourceV, 1, Indices_edgesS, Indices_edgesT);
else
    [error1, sourceV, transform] = rigidICP(targetV, sourceV, 0, Indices_edgesS, Indices_edgesT);
end

% plot
of
the
meshes
h = trisurf(sourceF, sourceV(:, 1), sourceV(:, 2), sourceV(:, 3), 0.3, 'Edgecolor', 'none');
hold
light
lighting
phong;
set(gca, 'visible', 'off')
set(gcf, 'Color', [1 1 0.88])
view(90, 90)
set(gca, 'DataAspectRatio', [1 1 1], 'PlotBoxAspectRatio', [1 1 1]);
tttt = trisurf(targetF, targetV(:, 1), targetV(:, 2), targetV(:, 3), 'Facecolor', 'm', 'Edgecolor', 'none');
alpha(0.6)

[p] = size(sourceV, 1);

% General
deformation
disp('General deformation');
kernel1 = 1.5:-(0.5 / iterations): 1;
kernel2 = 2.4:(0.3 / iterations): 2.7;
for i =1:iterations
nrseedingpoints = round(10 ^ (kernel2(1, i)));
IDX1 = [];
IDX2 = [];
[IDX1(:, 1), IDX1(:, 2)]=knnsearch(targetV, sourceV);
[IDX2(:, 1), IDX2(:, 2)]=knnsearch(sourceV, targetV);
IDX1(:, 3)=1: length(sourceV(:, 1));
IDX2(:, 3)=1: length(targetV(:, 1));


[C, ia] = setdiff(IDX1(:, 1), Indices_edgesT);
IDX1 = IDX1(ia,:);

[C, ia] = setdiff(IDX2(:, 1), Indices_edgesS);
IDX2 = IDX2(ia,:);


sourcepartial = sourceV(IDX1(:, 3),:);
targetpartial = targetV(IDX2(:, 3),:);

[IDXS, dS] = knnsearch(targetpartial, sourcepartial);
[IDXT, dT] = knnsearch(sourcepartial, targetpartial);

[ppartial] = size(sourcepartial, 1);
[tempF, tempV] = reducepatch(sourceF, sourceV, nrseedingpoints / size(sourceV, 1));
[idx] = knnsearch(sourcepartial, tempV);
[q] = size(idx, 1);

D = pdist2(sourcepartial, tempV);
gamma = 1 / (2 * (mean(mean(D))) ^ kernel1(1, i));

Datasetsource = vertcat(sourcepartial, sourcepartial(IDXT,:));

Datasettarget = vertcat(targetpartial(IDXS,:), targetpartial);
Datasetsource2 = vertcat(D, D(IDXT,:));
vectors = Datasettarget - Datasetsource;
[r] = size(vectors, 1);

% define
radial
basis
width
for deformation points
    % gaussian
tempy1 = exp(-gamma * (Datasetsource2. ^ 2));

tempy2 = zeros(3 * r, 3 * q);
tempy2(1: r, 1: q)=tempy1;
tempy2(r + 1: 2 * r, q + 1: 2 * q)=tempy1;
tempy2(2 * r + 1: 3 * r, 2 * q + 1: 3 * q)=tempy1;

% solve
optimal
deformation
directions
with regularisation term
% regularisation
lambda =0.001;
       ppi=inv((tempy2.'*tempy2)+lambda*eye(3*q))*(tempy2.');


       modes=ppi * reshape(vectors, 3 * r, 1);

       D2 = pdist2(sourceV, tempV);
gamma2 = 1 / (2 * (mean(mean(D2))) ^ kernel1(1, i));

tempyfull1 = exp(-gamma2 * (D2. ^ 2));
tempyfull2 = zeros(3 * p, 3 * q);
tempyfull2(1: p, 1: q)=tempyfull1;
tempyfull2(p + 1: 2 * p, q + 1: 2 * q)=tempyfull1;
tempyfull2(2 * p + 1: 3 * p, 2 * q + 1: 3 * q)=tempyfull1;

test2 = tempyfull2 * modes;
test2 = reshape(test2, size(test2, 1) / 3, 3);
% deforme
source
mesh
sourceV = sourceV + test2;

[error1, sourceV, transform] = rigidICP(targetV, sourceV, 1, Indices_edgesS, Indices_edgesT);
delete(h)
h = trisurf(sourceF, sourceV(:, 1), sourceV(:, 2), sourceV(:, 3), 'FaceColor', 'y', 'Edgecolor', 'none');
alpha(0.6)
pause(0.1)

end

% local
deformation
disp('Local optimization');
arraymap = repmat(cell(1), p, 1);
kk = 12 + iterations;

delete(tttt)
tttt = trisurf(targetF, targetV(:, 1), targetV(:, 2), targetV(:, 3), 'Facecolor', 'm', 'Edgecolor', 'none');

TR = triangulation(targetF, targetV);
normalsT = vertexNormal(TR). * cutoff;

% define
local
mesh
relation
TRS = triangulation(sourceF, sourceV);
normalsS = vertexNormal(TRS). * cutoff;
[IDXsource, Dsource] = knnsearch(horzcat(sourceV, normalsS), horzcat(sourceV, normalsS), 'K', kk);

% check
normal
direction
[IDXcheck, Dcheck] = knnsearch(targetV, sourceV);
testpos = sum(sum((normalsS - normalsT(IDXcheck,:)). ^ 2, 2));
testneg = sum(sum((normalsS + normalsT(IDXcheck,:)). ^ 2, 2));
if testneg < testpos
normalsT=-normalsT;
targetF(:,
    4)=targetF(:, 2);
targetF(:, 2)=[];
end

for ddd=1:iterations
k = kk - ddd;
tic

TRS = triangulation(sourceF, sourceV);
normalsS = vertexNormal(TRS). * cutoff;

sumD = sum(Dsource(:, 1: k), 2);
sumD2 = repmat(sumD, 1, k);
sumD3 = sumD2 - Dsource(:, 1: k);
sumD2 = sumD2 * (k - 1);
weights = sumD3. / sumD2;

[IDXtarget, Dtarget] = knnsearch(horzcat(targetV, normalsT), horzcat(sourceV, normalsS), 'K', 3);
pp1 = size(targetV, 1);

% correct
for holes in target
if isempty(Indices_edgesT) == 0

correctionfortargetholes1=find(ismember(IDXtarget(:,
    1), Indices_edgesT));
targetV = [targetV;
sourceV(correctionfortargetholes1,:)];
IDXtarget(correctionfortargetholes1, 1) = pp1 + (1:size(correctionfortargetholes1, 1))';
Dtarget(correctionfortargetholes1, 1) = 0.00001;

correctionfortargetholes2 = find(ismember(IDXtarget(:, 2), Indices_edgesT));
pp = size(targetV, 1);
targetV = [targetV;
sourceV(correctionfortargetholes2,:)];
IDXtarget(correctionfortargetholes2, 2) = pp + (1:size(correctionfortargetholes2, 1))';
Dtarget(correctionfortargetholes2, 2) = 0.00001;

correctionfortargetholes3 = find(ismember(IDXtarget(:, 3), Indices_edgesT));
pp = size(targetV, 1);
targetV = [targetV;
sourceV(correctionfortargetholes3,:)];
IDXtarget(correctionfortargetholes3, 3) = pp + (1:size(correctionfortargetholes3, 1))';
Dtarget(correctionfortargetholes3, 3) = 0.00001;

end

summD = sum(Dtarget, 2);
summD2 = repmat(summD, 1, 3);
summD3 = summD2 - Dtarget;
weightsm = summD3. / (summD2 * 2);
Targettempset = horzcat(weightsm(:, 1).*targetV(IDXtarget(:, 1), 1), weightsm(:, 1).*targetV(
    IDXtarget(:, 1), 2), weightsm(:, 1).*targetV(IDXtarget(:, 1), 3))+horzcat(weightsm(:, 2).*targetV(
    IDXtarget(:, 2), 1), weightsm(:, 2).*targetV(IDXtarget(:, 2), 2), weightsm(:, 2).*targetV(
    IDXtarget(:, 2), 3))+horzcat(weightsm(:, 3).*targetV(IDXtarget(:, 3), 1), weightsm(:, 3).*targetV(
    IDXtarget(:, 3), 2), weightsm(:, 3).*targetV(IDXtarget(:, 3), 3));

targetV = targetV(1:pp1,:);

arraymap = cell(size(sourceV, 1), 1);

for i = 1:size(sourceV, 1)
sourceset = sourceV(IDXsource(i, 1:k)',:);
targetset = Targettempset(IDXsource(i, 1:k)',:);
[d, z, arraymap
{i, 1}]=procrustes(targetset, sourceset, 'scaling', 0, 'reflection', 0);

end
sourceVapprox = sourceV;
for i = 1:size(sourceV, 1)
for ggg=1:k
sourceVtemp(ggg,:)=weights(i, ggg) * (arraymap{IDXsource(i, ggg), 1}.b * sourceV(i,:)*arraymap
{IDXsource(i, ggg), 1}.T + arraymap
{IDXsource(i, ggg), 1}.c(1,:));
end
sourceV(i,:)=sum(sourceVtemp(1: k,:));
end

sourceV = sourceVapprox + 0.5 * (sourceV - sourceVapprox);

toc
delete(h)
h = trisurf(sourceF, sourceV(:, 1), sourceV(:, 2), sourceV(:, 3), 'FaceColor', 'y', 'Edgecolor', 'none');
pause(0.1)

end

registered = sourceV;

extra.sourceV = sourceV;
extra.sourceF = sourceF;
extra.targetV = targetV;
extra.targetF = targetF;
"""


# # FUNCTION findClosestPoints()
# def findClosestPoints(point, Search_points, segmentation_radius, num_return_indices):
#     # TO BE DONE.
#     closest_indices = []
#     point = np.reshape(point, (1, 3))
#
#     aux_distances = distance.cdist(point, Search_points, 'sqeuclidean')[0]
#     closest_indices = np.argsort(aux_distances)
#
#     closest_indices = closest_indices[:num_return_indices]
#     return closest_indices

# FUNCTION findClosestPoints()
def findClosestPoints(point, Search_points, segmentation_radius, num_return_indices):
    # TO BE DONE.
    closest_indices = []
    if point.ndim == 1:
        point = np.reshape(point, (1, 3))

    num_points = point.shape[0]

    closest_indices = np.zeros((num_points, num_return_indices))

    if num_points == 1:
        aux_distances = distance.cdist(point, Search_points, 'euclidean')[0]
        aux_distances = np.reshape(aux_distances, (1, len(aux_distances)))
    else:
        aux_distances = distance.cdist(point, Search_points, 'euclidean')

    # print('num points: ' + str(num_points))

    # print('aux_distances shape: ' + str(aux_distances.shape))
    # print(aux_distances)
    for n in range(num_points):
        # print('n: ' + str(n))
        aux_closest_indices = np.argsort(aux_distances[n, :])
        # print(aux_closest_indices)
        closest_indices[n, :num_return_indices] = aux_closest_indices[:num_return_indices]

    if num_points == 1:
        closest_indices = closest_indices.squeeze()

    return closest_indices.astype(int)

def findClosestPointsByDistance(point, Search_points, segmentation_radius, num_return_indices, verbose=False):
    # TO BE DONE.
    closest_indices = []
    closest_distances = []
    if point.ndim == 1:
        point = np.reshape(point, (1, 3))

    num_points = point.shape[0]

    if verbose:
        aux_print = 'Num points to search: ' + str(int(num_points))
        print(aux_print)

    closest_indices = -1*np.ones((num_points, num_return_indices))
    closest_distances = -1 * np.ones((num_points, num_return_indices))

    if num_points == 1:
        aux_distances = distance.cdist(point, Search_points, 'euclidean')[0]
        aux_distances = np.reshape(aux_distances, (1, len(aux_distances)))
    else:
        aux_distances = distance.cdist(point, Search_points, 'euclidean')

    # print('num points: ' + str(num_points))

    # print('aux_distances shape: ' + str(aux_distances.shape))
    # print(aux_distances)
    for n in tqdm(range(num_points)):
        # print('n: ' + str(n))
        aux_closest_indices = np.argsort(aux_distances[n, :])
        # print(aux_closest_indices)
        aux_closest = aux_closest_indices[:num_return_indices]
        aux_sorted_distances = aux_distances[n, aux_closest_indices]

        if verbose:
            print(n, point[n,:])
            print(aux_sorted_distances, aux_closest)

        for d in range(num_return_indices):
            aux_dist = aux_sorted_distances[d]
            if aux_dist <= segmentation_radius:
                closest_indices[n, d] = aux_closest[d]
                closest_distances[n, d] = aux_dist
                aux_distances[:, aux_closest[d]] = float('inf')

    if num_points == 1:
        closest_indices = closest_indices.squeeze()
        closest_distances = closest_distances.squeeze()

    return closest_indices.astype(int), closest_distances

# FUNCTION calculateDensityMap
def calculateDensityMap(Points_on_map, Point_values, Vertices, Faces, p=0.5, normalize_map=False, show_figures=False, verbose=False):
    # TO BE DONE.

    num_vertices = Vertices.shape[0]
    DensityVertices = np.zeros((num_vertices, 1))
    # p = 2 # interpolation parameter

    if verbose:
        aux_print = 'Calculating density map'
        print(aux_print)

    for v in range(num_vertices):

        aux_point = Vertices[v, :]
        # aux_point = np.reshape(aux_point, (1,3))
        aux_density = 0

        aux_dist = distance.cdist(Points_on_map, np.reshape(aux_point, (1,3)), 'euclidean')
        aux_dist = np.power(aux_dist, p)
        aux_dist[aux_dist == 0] = 1
        aux_dist = np.reshape(aux_dist, (len(aux_dist),))
        aux_density = np.sum(Point_values/aux_dist)

        DensityVertices[v] = aux_density

    if normalize_map:
        if verbose:
            aux_print = 'Normalized density map'
            print(aux_print)

        DensityVertices = DensityVertices/np.max(DensityVertices)

    return DensityVertices

# FUNCTION calculateDensityMap
def calculateDensityMap2(Points_on_map, Point_values, Vertices, Search_distance=5, verbose=False):
    # TO BE DONE.

    total_points_sum = np.sum(Point_values)

    num_vertices = Vertices.shape[0]
    DensityVertices = np.zeros((num_vertices, 1))

    Mesh_to_points_distance = distance.cdist(Vertices, Points_on_map, 'euclidean')

    if verbose:
        aux_print = 'Calculating density map'
        print(aux_print)

    for v in range(num_vertices):

        aux_point = Vertices[v, :]
        aux_density = 0

        if verbose:
            if v % 1000 == 0:
                aux_percentage = np.round(100*v/num_vertices)
                aux_print = ' - Interpolating data: ' + str(aux_percentage) + '% (' + str(v) + '/' + str(Num_vertices) + ')'
                print(aux_print)

        aux_distance = Mesh_to_points_distance[v, :]
        aux_search_indices = np.where(aux_distance < Search_distance)[0]
        aux_search_values = Point_values[aux_search_indices]
        aux_density = np.sum(aux_search_values)/total_points_sum

        DensityVertices[v] = aux_density

    return DensityVertices

def interpolateDataPointsOnMesh(Data_points, Data_values, Vertices, Search_distance, split_distance=False, split_number=100, verbose=False):
                    # ALL_Analysis_voltage_points, ALL_Analysis_voltage_values, VerticesA, Search_distance = 5, verbose = False)

    Num_vertices = Vertices.shape[0]
    Interpolated_data = np.zeros((Num_vertices, 1))

    if split_distance:
        Mesh_to_points_distance = []

        partition_size = int(np.ceil(Num_vertices/split_number))
        for s in range(split_number):
            aux_print = 'Split ' + str(s) + '/' + str(split_number)
            print(aux_print)
            aux_init = 0+s*partition_size
            aux_end = partition_size+s*partition_size
            if aux_end > Num_vertices:
                aux_end = Num_vertices
            split_indices = np.arange(aux_init, aux_end, 1)

            aux_dist = distance.cdist(Vertices[split_indices, :], Data_points, 'euclidean')

            if verbose:
                aux_print = 'Interpolating data points on Mesh map'
                print(aux_print)

            for ii, v in enumerate(split_indices):

                if verbose:
                    if v % 1000 == 0:
                        aux_percentage = np.round(100 * v / Num_vertices)
                        aux_print = ' - Interpolating data: ' + str(aux_percentage) + '% (' + str(v) + '/' + str(
                            Num_vertices) + ')'
                        print(aux_print)

                aux_distance = aux_dist[ii, :]
                aux_search_indices = np.where(aux_distance < Search_distance)[0]
                aux_search_values = Data_values[aux_search_indices]
                if len(aux_search_values) > 0:

                    aux_interpolated_value = np.mean(aux_search_values)
                    Interpolated_data[v] = aux_interpolated_value

                else:
                    aux_interpolated_value = 0

                Interpolated_data[v] = aux_interpolated_value

    else:
        Mesh_to_points_distance = distance.cdist(Vertices, Data_points, 'euclidean')

        if verbose:
            aux_print = 'Interpolating data points on Mesh map'
            print(aux_print)


        for v in range(Num_vertices):

            if verbose:
                if v % 1000 == 0:
                    aux_percentage = np.round(100*v/Num_vertices)
                    aux_print = ' - Interpolating data: ' + str(aux_percentage) + '% (' + str(v) + '/' + str(Num_vertices) + ')'
                    print(aux_print)

            aux_distance = Mesh_to_points_distance[v, :]
            aux_search_indices = np.where(aux_distance < Search_distance)[0]
            aux_search_values = Data_values[aux_search_indices]
            if len(aux_search_values)>0:

                aux_interpolated_value = np.mean(aux_search_values)
                Interpolated_data[v] = aux_interpolated_value

            else:
                aux_interpolated_value = 0

            Interpolated_data[v] = aux_interpolated_value

    return Interpolated_data

def getSegmentationIndices(Vertices, Faces, Segmentation, LAA_expand_radius=0, PV_expand_radius=0, show_figures=False, merge_PV=False):

    Segmentation_indices = []
    # NEW SEGMENTATION (no PV merge)
    # 1 'LAA'       # 2 'PVs'
    # 3 'Posterior' # 4 'Roof'
    # 5 'Lateral'   # 6 'Septum'
    # 7 'Anterior'  # 8 'Floor'
    # 9 'PV Antra' # 10 'Mitral'

    # NEW SEGMENTATION (PV merge)
    # 1 'LAA'       # 2 'PV + PV Antra'
    # 3 'Posterior' # 4 'Roof'
    # 5 'Lateral'   # 6 'Septum'
    # 7 'Anterior'  # 8 'Floor'
    # 9 'Mitral'

    # SIMPLIFIED SEGMENTATION INDICES
    # 1 'LAA'       # 2 'PVs'
    # 3 'Posterior' # 4 'Roof'
    # 5 'Lateral'   # 6 'Septum'
    # 7 'Anterior'  # 8 'Floor'
    # 9 'Mitral'

    # ORIGINAL SEGMENTATION
    # 1 'Left roof'                         # 2 'Right roof'
    # 3 'Left superior posterior'           # 4 'Right superior posterior'
    # 5 'Left inferior posterior'           # 6 'Right inferior posterior'
    # 7 'Left floor'                        # 8 'Right floor'
    # 9 'Superior lateral'                  # 10 'Superior Septum'
    # 11 'Inferior lateral'                 # 12 'Inferior septum'
    # 13 'Left atrial appendage'            # 14 'Left superior pulmonary vein'
    # 15 'Right superior pulmonary vein'    # 16 'Left inferior pulmonary vein'
    # 17 'Right inferior pulmonary vein'    # 18 'Mitral Ring'
    # 19 'Extra'

    # ROOF 1, 2
    # POSTERIOR 3, 4, 5, 6
    # FLOOR 7, 8
    # LATERAL 9, 11
    # SEPTUM 10, 12
    # LAA 13
    # LSPV 14
    # RSPV 15
    # LIPV 16
    # RIPV 17
    # MITRAL RING 18
    # ANTERIOR 19
    Roof_indices = np.where((Segmentation == 1) | (Segmentation == 2))[0]
    Posterior_indices = np.where((Segmentation >=3) & (Segmentation <= 6))[0]
    Floor_indices = np.where((Segmentation == 7) | (Segmentation == 8))[0]
    Lateral_indices = np.where((Segmentation == 9) | (Segmentation == 11))[0]
    Septum_indices = np.where((Segmentation == 10) | (Segmentation == 12))[0]
    LAA_indices = np.where(Segmentation == 13)[0]
    LSPV_indices = np.where(Segmentation == 14)[0]
    RSPV_indices = np.where(Segmentation == 15)[0]
    LIPV_indices = np.where(Segmentation == 16)[0]
    RIPV_indices = np.where(Segmentation == 17)[0]
    Mitral_indices =np.where(Segmentation == 18)[0]
    Anterior_indices = np.where(Segmentation == 19)[0]
    LPVs_indices =  np.where((Segmentation == 14) | (Segmentation == 16))[0]
    RPVs_indices = np.where((Segmentation == 15) | (Segmentation == 17))[0]

    LAA_expanded_indices = expandSegmentedIndicesByLabel(Vertices, Faces, Segmentation, 13, LAA_expand_radius)
    laa_lvps_indices = np.intersect1d(LAA_expanded_indices, LPVs_indices)
    LAA_expanded_indices = np.setdiff1d(LAA_expanded_indices, laa_lvps_indices)

    LSPV_expanded_indices = expandSegmentedIndicesByLabel(Vertices, Faces, Segmentation, 14, PV_expand_radius)
    RSPV_expanded_indices = expandSegmentedIndicesByLabel(Vertices, Faces, Segmentation, 15, PV_expand_radius)
    LIPV_expanded_indices = expandSegmentedIndicesByLabel(Vertices, Faces, Segmentation, 16, PV_expand_radius)
    RIPV_expanded_indices = expandSegmentedIndicesByLabel(Vertices, Faces, Segmentation, 17, PV_expand_radius)

    # PULMONARY VEINS INDICES
    LPVs_expanded_indices = np.hstack((LSPV_expanded_indices, LIPV_expanded_indices))
    LPVs_expanded_indices = np.unique(LPVs_expanded_indices)
    RPVs_expanded_indices = np.hstack((RSPV_expanded_indices, RIPV_expanded_indices))
    RPVs_expanded_indices = np.unique(RPVs_expanded_indices)
    lpvs_laa_indices = np.intersect1d(LAA_expanded_indices, LPVs_expanded_indices)
    LPVs_expanded_indices = np.setdiff1d(LPVs_expanded_indices, lpvs_laa_indices)

    # PULMONARY VEINS ANTRA
    PV_right_antrum_indices = np.setdiff1d(RPVs_expanded_indices, RPVs_indices)
    PV_left_antrum_indices = np.setdiff1d(LPVs_expanded_indices, LPVs_indices)

    # POSTERIOR INDICES
    post_laa_indices = np.intersect1d(Posterior_indices, LAA_expanded_indices)
    Posterior_indices = np.setdiff1d(Posterior_indices, post_laa_indices)
    post_lpvs_indices = np.intersect1d(Posterior_indices, LPVs_expanded_indices)
    Posterior_indices = np.setdiff1d(Posterior_indices, post_lpvs_indices)
    post_rpvs_indices = np.intersect1d(Posterior_indices, RPVs_expanded_indices)
    Posterior_indices = np.setdiff1d(Posterior_indices, post_rpvs_indices)

    # ROOF INDICES
    roof_laa_indices = np.intersect1d(Roof_indices, LAA_expanded_indices)
    Roof_indices = np.setdiff1d(Roof_indices, roof_laa_indices)
    roof_lpvs_indices = np.intersect1d(Roof_indices, LPVs_expanded_indices)
    Roof_indices = np.setdiff1d(Roof_indices, roof_lpvs_indices)
    roof_rpvs_indices = np.intersect1d(Roof_indices, RPVs_expanded_indices)
    Roof_indices = np.setdiff1d(Roof_indices, roof_rpvs_indices)

    # FLOOR INDICES
    floor_laa_indices = np.intersect1d(Floor_indices, LAA_expanded_indices)
    Floor_indices = np.setdiff1d(Floor_indices, floor_laa_indices)
    floor_lpvs_indices = np.intersect1d(Floor_indices, LPVs_expanded_indices)
    Floor_indices = np.setdiff1d(Floor_indices, floor_lpvs_indices)
    floor_rpvs_indices = np.intersect1d(Floor_indices, RPVs_expanded_indices)
    Floor_indices = np.setdiff1d(Floor_indices, floor_rpvs_indices)

    # LATERAL INDICES
    lateral_laa_indices = np.intersect1d(Lateral_indices, LAA_expanded_indices)
    Lateral_indices = np.setdiff1d(Lateral_indices, lateral_laa_indices)
    lateral_lpvs_indices = np.intersect1d(Lateral_indices, LPVs_expanded_indices)
    Lateral_indices = np.setdiff1d(Lateral_indices, lateral_lpvs_indices)
    lateral_rpvs_indices = np.intersect1d(Lateral_indices, RPVs_expanded_indices)
    Lateral_indices = np.setdiff1d(Lateral_indices, lateral_rpvs_indices)

    # SEPTUM INDICES
    septum_laa_indices = np.intersect1d(Septum_indices, LAA_expanded_indices)
    Septum_indices = np.setdiff1d(Septum_indices, septum_laa_indices)
    septum_lpvs_indices = np.intersect1d(Septum_indices, LPVs_expanded_indices)
    Septum_indices = np.setdiff1d(Septum_indices, septum_lpvs_indices)
    septum_rpvs_indices = np.intersect1d(Septum_indices, RPVs_expanded_indices)
    Septum_indices = np.setdiff1d(Septum_indices, septum_rpvs_indices)

    # MITRAL RING
    mitral_laa_indices = np.intersect1d(Mitral_indices, LAA_expanded_indices)
    Mitral_indices = np.setdiff1d(Mitral_indices, mitral_laa_indices)
    mitral_lpvs_indices = np.intersect1d(Mitral_indices, LPVs_expanded_indices)
    Mitral_indices = np.setdiff1d(Mitral_indices, mitral_lpvs_indices)
    mitral_rpvs_indices = np.intersect1d(Mitral_indices, RPVs_expanded_indices)
    Mitral_indices = np.setdiff1d(Mitral_indices, mitral_rpvs_indices)

    # ANTERIOR
    anterior_laa_indices = np.intersect1d(Anterior_indices, LAA_expanded_indices)
    Anterior_indices = np.setdiff1d(Anterior_indices, anterior_laa_indices)
    anterior_lpvs_indices = np.intersect1d(Anterior_indices, LPVs_expanded_indices)
    Anterior_indices = np.setdiff1d(Anterior_indices, anterior_lpvs_indices)
    anterior_rpvs_indices = np.intersect1d(Anterior_indices, RPVs_expanded_indices)
    Anterior_indices = np.setdiff1d(Anterior_indices, anterior_rpvs_indices)

    # SEGMENTATION INDICES
    Segmentation_indices.append(LAA_expanded_indices)
    # Segmentation_indices.append(LPVs_expanded_indices)
    # Segmentation_indices.append(RPVs_expanded_indices)
    Segmentation_indices.append(LSPV_indices)
    Segmentation_indices.append(RSPV_indices)
    Segmentation_indices.append(LIPV_indices)
    Segmentation_indices.append(RIPV_indices)
    if merge_PV:
        Segmentation_indices.append(PV_left_antrum_indices)
        Segmentation_indices.append(PV_right_antrum_indices)
    Segmentation_indices.append(Posterior_indices)
    Segmentation_indices.append(Roof_indices)
    Segmentation_indices.append(Lateral_indices)
    Segmentation_indices.append(Septum_indices)
    Segmentation_indices.append(Anterior_indices)
    Segmentation_indices.append(Floor_indices)
    if merge_PV == False:
        Segmentation_indices.append(PV_left_antrum_indices)
        Segmentation_indices.append(PV_right_antrum_indices)
    Segmentation_indices.append(Mitral_indices)

    # SEGMENTATION NAMES
    if merge_PV:
        Segmentation_names = ['LAA', 'PVs + PV Antra', 'Posterior', 'Roof', 'Lateral', 'Septum', 'Anterior', 'Floor', 'Mitral']
    else:
        Segmentation_names = ['LAA', 'PVs', 'Posterior', 'Roof', 'Lateral', 'Septum', 'Anterior', 'Floor', 'PV Antra', 'Mitral']

    # SIMPLIFIED SEGMENTATION
    Simplified_Segmentation = -1*np.ones(Segmentation.shape)
    Simplified_Segmentation[LAA_expanded_indices] = 1
    # Simplified_Segmentation[LPVs_expanded_indices] = 2
    # Simplified_Segmentation[RPVs_expanded_indices] = 2
    Simplified_Segmentation[LSPV_indices] = 2
    Simplified_Segmentation[RSPV_indices] = 2
    Simplified_Segmentation[LIPV_indices] = 2
    Simplified_Segmentation[RIPV_indices] = 2
    if merge_PV:
        Simplified_Segmentation[PV_left_antrum_indices] = 2
        Simplified_Segmentation[PV_left_antrum_indices] = 2
        Simplified_Segmentation[Mitral_indices] = 9
    else:
        Simplified_Segmentation[PV_left_antrum_indices] = 9
        Simplified_Segmentation[PV_right_antrum_indices] = 9
        Simplified_Segmentation[Mitral_indices] = 10

    Simplified_Segmentation[Posterior_indices] = 3
    Simplified_Segmentation[Roof_indices] = 4
    Simplified_Segmentation[Lateral_indices] = 5
    Simplified_Segmentation[Septum_indices] = 6
    Simplified_Segmentation[Anterior_indices] = 7
    Simplified_Segmentation[Floor_indices] = 8

    # CHECK IF THERE ARE UNASSIGNED ELEMENTS
    num_vertices = Vertices.shape[0]
    unassigned_indices = np.where(Simplified_Segmentation==-1)[0]
    unassigned_vertices = Vertices[unassigned_indices, :]
    assigned_indices = np.arange(num_vertices)
    assigned_indices = np.setdiff1d(assigned_indices, unassigned_indices)
    assigned_vertices = Vertices[assigned_indices, :]

    num_unassigned_vertices = len(unassigned_indices)
    for v in range(num_unassigned_vertices):
        aux_pos = unassigned_vertices[v, :]
        aux_dist = (assigned_vertices - aux_pos) ** 2
        aux_dist = np.sum(aux_dist, axis=1)
        aux_dist = np.sqrt(aux_dist)
        aux_min_pos = np.argmin(aux_dist)
        aux_a_index = assigned_indices[aux_min_pos]
        aux_u_index = unassigned_indices[v]
        aux_segmentation_value = Simplified_Segmentation[aux_a_index]
        Simplified_Segmentation[aux_u_index] = aux_segmentation_value


    if show_figures:
        # aux_plot_mesh = np.zeros(Segmentation.shape)
        # aux_plot_mesh[LAA_expanded_indices] = 1
        # aux_plot_mesh[LPVs_expanded_indices] = 2
        # aux_plot_mesh[RPVs_expanded_indices] = 3
        # aux_plot_mesh[Posterior_indices] = 4
        # aux_plot_mesh[Roof_indices] = 5
        # aux_plot_mesh[Lateral_indices] = 6
        # aux_plot_mesh[Septum_indices] = 7
        # aux_plot_mesh[Mitral_indices] = 8
        # aux_plot_mesh[Anterior_indices] = 9
        plotMesh(Vertices, Faces, Simplified_Segmentation, [np.min(Simplified_Segmentation), np.max(Simplified_Segmentation)])

        # plt.figure()
        # plt.hist(Simplified_Segmentation, bins=50)
        # plt.show()

        Simplified_Segmentation = np.squeeze(Simplified_Segmentation, axis=1)

    return Segmentation_indices, Segmentation_names, Simplified_Segmentation

def expandSegmentedIndicesByLabel(Vertices, Triangles, Segmentation, Expansion_label, Expand_radius):

    # Get label indices
    aux_segmented_indices = np.where(Segmentation == Expansion_label)[0]
    aux_segmented_vertices = Vertices[aux_segmented_indices, :]

    keep_expanding = True
    expanded_indices = np.unique(aux_segmented_indices)

    search_index = 0

    while(keep_expanding):

        num_expanded_indices = len(expanded_indices)
        next_search_index = num_expanded_indices
        aux_search_indices = np.arange(search_index, num_expanded_indices, 1)

        for i, aux_i in enumerate(aux_search_indices):

            aux_index = expanded_indices[aux_i]

            # Find neighbouring vertices
            aux_Tx = np.where(Triangles[:, 0] == aux_index)[0]
            aux_Ty = np.where(Triangles[:, 1] == aux_index)[0]
            aux_Tz = np.where(Triangles[:, 2] == aux_index)[0]

            aux_new_triangle_indices = np.hstack((aux_Tx, aux_Ty, aux_Tz))
            aux_new_triangle_indices = np.unique(aux_new_triangle_indices)

            aux_neighbours = np.empty((0,)).astype(int)
            for n, t_index in enumerate(aux_new_triangle_indices):
                aux_vertices = Triangles[t_index, :]
                aux_neighbours = np.hstack((aux_neighbours, aux_vertices))

            aux_neighbours = np.unique(aux_neighbours)

            # Check if they are already in the expanded indices array
            aux_new_neighbours = np.in1d(aux_neighbours, expanded_indices)
            aux_new_neighbours = np.logical_not(aux_new_neighbours)
            aux_new_neighbours = aux_neighbours[aux_new_neighbours]

            if len(aux_new_neighbours)>0:

                # print(aux_i)

                # Check if they are within the desired radius wrt the Label region
                aux_in_search_range = np.zeros(aux_new_neighbours.shape, dtype=bool)
                for n, aux_p_index in enumerate(aux_new_neighbours):
                    aux_point = Vertices[aux_p_index, :]
                    aux_indices, aux_distances = findClosestPointsByDistance(aux_point, aux_segmented_vertices, Expand_radius, 1)

                    if aux_indices.ndim == 0:
                        if aux_indices > -1:
                            aux_in_search_range[n] = True

                aux_new_neighbours = aux_new_neighbours[aux_in_search_range]

                # Add new vertices to the expanded array
                expanded_indices = np.hstack((expanded_indices, aux_new_neighbours))

        # Stop condition
        if len(expanded_indices) == num_expanded_indices:
            keep_expanding = False
        else:
            search_index = next_search_index

    return np.unique(expanded_indices)