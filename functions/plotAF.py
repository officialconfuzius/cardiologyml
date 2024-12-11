"""Help on built-in module plotAF:

NAME
    plotAF

DESCRIPTION
    This module provides the necessary functions to plot all kind of AF related signals

CLASSES
    ...
    ...

FUNCTIONS
    plotBipolarEGMs(bipolar_signals)
        Plots bipolar EGMs. Size [Nb, L]
        Returns None


"""
import numpy as np
import pandas as pd
from functions.importOBJ_functions import *
import chart_studio.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
# py.offline.init_notebook_mode()
import plotly.io as pio
pio.renderers.default = 'browser'
from math import pi
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest


#-------------------------------------------------------------------------------------------------
# CLASSES
#-------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------------------------
def plotBipolarEGMs(bipolar_signals, plot_xlim=[0, 2000], figure_title=''):
    """
    Function: plotBipolarEGMs(bipolar_signals)

    Parameters:
        bipolar_signals (numpy float array) : Bipolar EGM signals. Size [Nu, L]
        plot_xlim (float array) : limits for the x-axis
    Returns:
        None
   """
    [Nb, L] = bipolar_signals.shape

    R = np.linspace(0, 1, Nb + 1)
    # color = plt.cm.hsv(R)

    # plt.figure()
    plot_offset = -1
    line_width = 2
    plot_ylim = [-Nb - 1, 0]

    fig = go.Figure()

    # aux_B = np.zeros(bipolar_signals.shape)
    for b in range(Nb):
        aux_bipolar_signal = bipolar_signals[b, :]
        aux_bipolar_signal = aux_bipolar_signal + (b+1)*plot_offset

        aux_name = 'B' + str(b+1)

        fig.add_trace(go.Scatter(
            x=np.linspace(0, L, L),
            y=aux_bipolar_signal,
            # fill='toself',
            # fillcolor='rgba(0,100,80,0.2)',
            # line_color='rgba(255,255,255,0)',
            # showlegend=False,
            name=aux_name
        ))
        # aux_B[b,:] = aux_bipolar_signal
        # plt.plot(aux_bipolar_signal, color=color[b, :], label='s' + str(b), linewidth=line_width)

    fig.update_layout(title_text=figure_title, title_x=0.5,
                      xaxis_title="Time (ms)",
                      yaxis_title="EGM index",
                      xaxis_range=plot_xlim,
                      yaxis_range=plot_ylim,
                      )
    fig.show()

# FUNCTION plotCartoMeshData()
def plotCartoMeshData(CartoMeshData, clim_limits, figure_title='Mesh plot'):
    # TO BE DONE

    x = CartoMeshData.X
    y = CartoMeshData.Y
    z = CartoMeshData.Z
    v0 = CartoMeshData.Vertex0
    v1 = CartoMeshData.Vertex1
    v2 = CartoMeshData.Vertex2
    triangles = np.vstack((v0, v1, v2)).T
    num_triangles = CartoMeshData.NumTriangle

    bipolar = CartoMeshData.Bipolar
    if len(bipolar == 0):
        if len(bipolar) == num_triangles:
            color_func = bipolar
        else:
            # Color data is referred to each vertex, the plot function takes color referred to the triangles
            color_func = np.zeros((num_triangles,))
            for t in range(num_triangles):
                aux_triangle = triangles[t, :]
                aux_color_values = bipolar[aux_triangle]
                color_func[t] = np.mean(aux_color_values)

            # Colorbar range
            aux_min_color = np.where(color_func < clim_limits[0])
            aux_max_color = np.where(color_func > clim_limits[1])

            print(aux_min_color)
            print(aux_max_color)
            color_func[aux_min_color] = clim_limits[0]
            color_func[aux_max_color] = clim_limits[1]
    else:
        color_func = None

    fig = ff.create_trisurf(x=x, y=y, z=z,
                            colormap="Portland",
                            simplices=triangles,
                            title=figure_title,
                            plot_edges=False,
                            color_func=color_func)

    fig.show()


    # Creating figure
    print('aaaaaaaaaaaaaaaaaaaaaaa')
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection='3d')

    # Creating color map
    my_cmap = plt.get_cmap('hot')

    # Creating plot
    trisurf = ax.plot_trisurf(x, y, z,
                              cmap=my_cmap,
                              linewidth=0.2,
                              antialiased=True,
                              edgecolor='grey')
    fig.colorbar(trisurf, ax=ax, shrink=0.5, aspect=5)
    ax.set_title('Tri-Surface plot')

    # Adding labels
    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Y-axis', fontweight='bold')
    ax.set_zlabel('Z-axis', fontweight='bold')

    # show plot
    plt.show()

# FUNCTION plotMesh()
def plotMesh(vertices, triangles, triangles_color, clim_limits, figure_title='Mesh plot', fig='', save_figure_path='', row='', col='', scene_name='', show_grid=True, hover_title='', opacity=1):
    # TO BE DONE
    num_triangles = triangles.shape[0]

    x = vertices[:,0]
    y = vertices[:,1]
    z = vertices[:,2]

    aux_x_range = np.max(x) - np.min(x)
    aux_y_range = np.max(y) - np.min(y)
    aux_z_range = np.max(z) - np.min(z)
    aux_x_aspect = aux_x_range/aux_x_range
    aux_y_aspect = aux_y_range / aux_x_range
    aux_z_aspect = aux_z_range / aux_x_range

    aux_intensity = np.copy(triangles_color)
    aux_intensity[aux_intensity>clim_limits[1]] = clim_limits[1]
    aux_intensity[aux_intensity < clim_limits[0]] = clim_limits[0]

    # hove data
    aux_customdata = np.stack((x, y, z, triangles_color), axis=-1)
    aux_hovertemplate = "[x,y,z]: %{customdata[0]:.2f}, %{customdata[1]:.2f}, %{customdata[2]:.2f}"+\
                             "<br><b>" + hover_title + "</b>: %{customdata[3]:.2f}<extra></extra>"

    if fig == '':
        fig = go.Figure(layout_title_text=figure_title)

    trace_mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        # colorbar_title='z',
        # colorscale=[[0, 'gold'],
        #             [0.5, 'mediumturquoise'],
        #             [1, 'magenta']],
        colorscale='rainbow',
        reversescale=True,
        opacity=opacity,
        showlegend=True,
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=aux_intensity,
        cmin=clim_limits[0],
        cmax=clim_limits[1],
        # i, j and k give the vertices of triangles
        # here we represent the 4 triangles of the tetrahedron surface
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        name='mesh',
        showscale=True,
        customdata=aux_customdata,
        hovertemplate=aux_hovertemplate
    )

    if row == '':
        fig.add_trace(trace_mesh)
    else:
        fig.add_trace(trace_mesh, row=row, col=col)

    fig.update_layout(title_text=figure_title, title_x=0.5)

    if show_grid == False:
        fig.update_layout(scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white", ),
            yaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white"),
            zaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white", ), )
        )

    fig.show()

    if save_figure_path != '':
        fig.write_html(save_figure_path)

    return fig



def plotMeshTransformation(VerticesA, FacesA, VerticesB, FacesB, VerticesB_on_A, skip_indices=10, show_mesh=True, show_vertices=True, show_transformation=True, transformation_lines_percentage=0.01):

    figure_title = 'MESH B ON A TRANSFORMATION'
    aux_x_aspect = 1
    aux_y_aspect = 1
    aux_z_aspect = 1
    color_func = None

    fig = go.Figure()

    trace_B = go.Scatter3d(x=VerticesB[0:VerticesB.shape[0]:skip_indices, 0],
                                       y=VerticesB[0:VerticesB.shape[0]:skip_indices, 1],
                                       z=VerticesB[0:VerticesB.shape[0]:skip_indices, 2],
                                       name='MeshB points',
                                       mode='markers', marker=dict(
            size=4,
            color='red',  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        ))
    trace_A = go.Scatter3d(x=VerticesA[0:VerticesA.shape[0]:skip_indices, 0],
                           y=VerticesA[0:VerticesA.shape[0]:skip_indices, 1],
                           z=VerticesA[0:VerticesA.shape[0]:skip_indices, 2],
                           name='MeshA points',
                           mode='markers', marker=dict(
            size=4,
            color='green',  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        ))
    trace_B_on_A = go.Scatter3d(x=VerticesB_on_A[0:VerticesB_on_A.shape[0]:skip_indices, 0],
                           y=VerticesB_on_A[0:VerticesB_on_A.shape[0]:skip_indices, 1],
                           z=VerticesB_on_A[0:VerticesB_on_A.shape[0]:skip_indices, 2],
                           name='MeshB points on A',
                           mode='markers', marker=dict(
            size=4,
            color='blue',  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        ))


    trace_meshA = go.Mesh3d(
        x=VerticesA[:,0],
        y=VerticesA[:,1],
        z=VerticesA[:,2],
        # colorbar_title='z',
        colorscale=[[0, 'gold'],
                    [0.5, 'mediumturquoise'],
                    [1, 'magenta']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=[0, 0.33, 0.66, 1],
        # i, j and k give the vertices of triangles
        # here we represent the 4 triangles of the tetrahedron surface
        i=FacesA[:,0],
        j=FacesA[:,1],
        k=FacesA[:,2],
        name='meshA',
        showscale=True
    )

    trace_meshB = go.Mesh3d(
        x=VerticesB[:, 0],
        y=VerticesB[:, 1],
        z=VerticesB[:, 2],
        # colorbar_title='z',
        # colorscale=[[0, 'gold'],
        #             [0.5, 'mediumturquoise'],
        #             [1, 'magenta']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=[0, 0.33, 0.66, 1],
        # i, j and k give the vertices of triangles
        # here we represent the 4 triangles of the tetrahedron surface
        i=FacesB[:, 0],
        j=FacesB[:, 1],
        k=FacesB[:, 2],
        name='meshB',
        showscale=True
    )

    if show_mesh:
        fig.add_trace(trace_meshA)
        fig.add_trace(trace_meshB)

    if show_vertices:
        fig.add_trace(trace_B)
        fig.add_trace(trace_A)
        fig.add_trace(trace_B_on_A)

    if show_transformation:
        num_transformation_lines = int(transformation_lines_percentage*VerticesB.shape[0])
        aux_line_indices = np.linspace(0, VerticesB.shape[0], num_transformation_lines).astype(int)
        aux_line_indices = aux_line_indices[:-1]
        for n, line_index in enumerate(aux_line_indices):
            aux_x0 = VerticesB[line_index, 0]
            aux_y0 = VerticesB[line_index, 1]
            aux_z0 = VerticesB[line_index, 2]
            aux_x1 = VerticesB_on_A[line_index, 0]
            aux_y1 = VerticesB_on_A[line_index, 1]
            aux_z1 = VerticesB_on_A[line_index, 2]
            aux_trace_line = go.Scatter3d(
                x=[aux_x0, aux_x1], y=[aux_y0, aux_y1], z=[aux_z0, aux_z1],
                marker=dict(
                    size=4,
                    colorscale='Viridis',
                ),
                line=dict(
                    color='darkblue',
                    width=2
                ))

            fig.add_trace(aux_trace_line)

    fig.show()

def plotCartoFinderPointsOnMesh(VerticesA, FacesA, CartoFinderPoints, figure_title='', CartoFinderPoints_color='blue',
                                intensity=[0, 0.33, 0.66, 1], show_grid=True, opacity=1,
                                legend_title_mesh='', legend_title_scatter='', clim_scatter=[0, 1], scatter_size=100,
                                colorscale_mesh='rainbow', invert_colorbar_mesh=False,
                                colorscale_scatter='RdBu', invert_colorbar_scatter=False,
                                mesh_name='', scatter_name='',
                                hover_info=[], hover_txt_str='',
                                save_figure=False, save_figure_path='',
                                clim_mesh=[],
                                plotCatheter=False, CatheterModel='PentaRay', CatheterPositions='All', catheter_color='black',
                                show_axis=True):
    if invert_colorbar_mesh:
        colorscale_mesh = colorscale_mesh + '_r'
    if invert_colorbar_scatter:
        colorscale_scatter = colorscale_scatter + '_r'

    if CartoFinderPoints.ndim == 1:
        CartoFinderPoints = np.reshape(CartoFinderPoints, (1,3))

    if clim_mesh == []:
        clim_mesh = [0, 0]
        clim_mesh[0] = intensity.min()
        clim_mesh[1] = intensity.max()

    fig = go.Figure(layout_title_text=figure_title)

    trace_meshA = go.Mesh3d(
        x=VerticesA[:, 0],
        y=VerticesA[:, 1],
        z=VerticesA[:, 2],
        showlegend=True,
        hoverinfo='text',
        # colorbar_title='z',
        colorscale=colorscale_mesh,
        cmin=clim_mesh[0],
        cmax=clim_mesh[1],
        reversescale=True,
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=intensity,
        opacity=opacity,
        # i, j and k give the vertices of triangles
        # here we represent the 4 triangles of the tetrahedron surface
        i=FacesA[:, 0],
        j=FacesA[:, 1],
        k=FacesA[:, 2],
        name=mesh_name,
        showscale=True,
        colorbar=dict(title=legend_title_mesh, thicknessmode="pixels", thickness=15,
                      x=1, titleside='right')
    )

    aux_num_points = CartoFinderPoints.shape[0]
    if len(hover_info)>0:
        aux_hovertext = [(hover_txt_str + ' ' + str(int(aux_val)) + ', ' + legend_title_scatter + ': ' + "{:.2f}".format(CartoFinderPoints_color[a])) for a, aux_val in enumerate(hover_info)]
    else:
        aux_hovertext = [('P: [' + "{:.2f}".format(CartoFinderPoints[a,0]) + ', ' + "{:.2f}".format(CartoFinderPoints[a,1]) + ', ' + "{:.2f}".format(CartoFinderPoints[a,2]) + ']') for a in range(aux_num_points)]

    trace_A = go.Scatter3d(x=CartoFinderPoints[:,0],
                           y=CartoFinderPoints[:,1],
                           z=CartoFinderPoints[:,2],
                           hovertext=aux_hovertext,
                           showlegend=True,
                           name=scatter_name,
                           mode='markers', marker=dict(
            size=scatter_size,
            color=CartoFinderPoints_color,  # set color to an array/list of desired values
            colorscale=colorscale_scatter,  # choose a colorscale
            opacity=0.8,
            cmin=clim_scatter[0],
            cmax=clim_scatter[1],
            colorbar=dict(title=legend_title_scatter, thicknessmode="pixels", thickness=15,
                          x=1.05, titleside='right')
        ))

    fig.add_trace(trace_meshA)
    fig.add_trace(trace_A)
    trace_meshA.opacity = 0.05
    fig.add_trace(trace_meshA)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    if show_grid == False:
        fig.update_layout(scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white", ),
            yaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white"),
            zaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white", ), )
        )

    if plotCatheter:
        # Plot Catheter Splines
        if CatheterModel == 'PentaRay':
            Nu = 20
            Nb = 15
            num_splines = 5
            num_electrode_per_spline = 4

            if CatheterPositions == 'All':
                NumCatheterPositions = int(CartoFinderPoints.shape[0]/Nu)
            else:
                NumCatheterPositions = len(CatheterPositions)

            for n in range(NumCatheterPositions):

                aux_i = CatheterPositions[n]
                aux_catheter_positions = np.arange(aux_i*Nu, (aux_i+1)*Nu)
                aux_catheter_positions = aux_catheter_positions.astype(int)
                aux_catheter_positions = aux_catheter_positions[aux_catheter_positions < CartoFinderPoints.shape[0]]

                aux_catheter_points = CartoFinderPoints[aux_catheter_positions, :]
                aux_catheter_points_plot = np.empty((0, 3))
                aux_nan_point = np.zeros((1, 3))
                aux_nan_point[:] = np.nan

                for s in range(num_splines):
                    aux_spline_indices = np.arange(s*num_electrode_per_spline, (s+1)*num_electrode_per_spline)
                    print(aux_spline_indices)

                    # Stack the points
                    aux_catheter_points_plot = np.vstack((aux_catheter_points_plot, aux_catheter_points[aux_spline_indices, :]))
                    # Stack nan point
                    aux_catheter_points_plot = np.vstack((aux_catheter_points_plot,aux_nan_point))

                aux_catheter_trace = go.Scatter3d(x=aux_catheter_points_plot[:, 0],
                                                  y=aux_catheter_points_plot[:, 1],
                                                  z=aux_catheter_points_plot[:, 2],
                                                  showlegend=True,
                                                  visible="legendonly",
                                                  name='Catheter ' + str(aux_i),
                                                  mode='lines', line=dict(
                        color=catheter_color,  # set color to an array/list of desired values
                        width=10,
                    ))

                fig.add_trace(aux_catheter_trace)

                scatter_name = 'Catheter ' + str(aux_i) + ' values'
                aux_catheter_color = CartoFinderPoints_color[aux_catheter_positions]
                #aux_catheter_color = np.reshape(aux_catheter_color, (aux_catheter_color.shape[0], 1))

                trace_catheter_color = go.Scatter3d(x=aux_catheter_points[:, 0],
                                       y=aux_catheter_points[:, 1],
                                       z=aux_catheter_points[:, 2],
                                       hovertext=aux_hovertext,
                                       showlegend=True,
                                       visible="legendonly",
                                       name=scatter_name,
                                       mode='markers', marker=dict(
                        size=scatter_size,
                        color=aux_catheter_color,  # set color to an array/list of desired values
                        colorscale=colorscale_scatter,  # choose a colorscale
                        opacity=0.8,
                        cmin=clim_scatter[0],
                        cmax=clim_scatter[1],
                        colorbar=dict(title=legend_title_scatter, thicknessmode="pixels", thickness=15,
                                      x=1.05, titleside='right')
                    ))

                fig.add_trace(trace_catheter_color)

    if show_axis == False:
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )

    fig.update_layout(title_text=figure_title, title_x=0.5)
    fig.show()

    if save_figure and save_figure_path != '':
        fig.write_html(save_figure_path)


# def plotEntanglement(Bipolar_signals, Entanglement, plot_xlim=[0, 5000], cmap='Reds', rings_order=False, CFData=None):
#
#     aux_linewidth = 0.1
#
#     bipolar_rings_indices = [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
#
#     Bipolar_signals_plot = np.copy(Bipolar_signals)
#     Entanglement_plot = np.copy(Entanglement)
#
#     entanglement_title = 'Entanglement'
#
#     if rings_order:
#         Bipolar_signals_plot = Bipolar_signals_plot[bipolar_rings_indices, :]
#         Entanglement_plot = Entanglement_plot[bipolar_rings_indices, :]
#         entanglement_title = entanglement_title + ' - Rings order'
#
#     # plot_xlim = [10000, 15000]
#
#     [Nb, L] = Bipolar_signals_plot.shape
#
#     plt.figure(figsize=(18, 12))
#     plt.ion()
#     aux_offset = 0
#     aux_half_offset = 0
#
#     for n in range(Nb):
#         aux_egm = Bipolar_signals_plot[n, :] + aux_offset + aux_half_offset
#         aux_label = 'B' + str(n)
#         plt.plot(aux_egm, linewidth=aux_linewidth, label=aux_label)
#         aux_offset = aux_offset + 1
#
#     if CFData != None:
#
#         cf_start_time = CFData.StartTime
#         cf_events = CFData.Events
#         num_events = len(cf_events)
#
#         for n in range(num_events):
#             aux_event = cf_events[n]
#             if aux_event.Rotor == 1:
#                 aux_init = aux_event.EventInit - cf_start_time
#                 aux_end = aux_event.EventEnd - cf_start_time
#                 aux_t = np.arange(aux_init,aux_end)
#                 aux_offset = 0
#                 for b in range(Nb):
#                     aux_egm = Bipolar_signals_plot[b, aux_init:aux_end] + aux_offset + aux_half_offset
#                     plt.plot(aux_t, aux_egm, 'k', linewidth=aux_linewidth)
#                     aux_offset = aux_offset + 1
#
#
#
#     plt.imshow(Entanglement_plot, interpolation='None', aspect='auto', cmap=cmap)
#
#     plt.legend()
#
#     plt.xlim(plot_xlim)
#     plt.ylim([-1, Nb+1])
#     plt.show()
#     plt.colorbar()
#     plt.title(entanglement_title)
#
#     # See if this works...
#     plt.draw()
#     plt.pause(0.001)
#
#     plt.figure()
#     plt.ion()
#     plt.imshow(Entanglement_plot, interpolation='None', aspect='auto', cmap=cmap)
#     plt.xlim(plot_xlim)
#     plt.colorbar()
#     plt.show()
#
#     # See if this works...
#     plt.draw()
#     plt.pause(0.001)

# Rotor_Areas_per_patient[:aux_gender_len, :-1], aux_male_indices, aux_female_indices, Segmentation_names[:-1], aux_title
def plotSpiderChart(info_matrix, indices_male, indices_female, categories, figure_title='', y_pos=[25, 50], y_pos_str=["25%", "50%"], ylim_max=50, percentage=False):
    ## CREATE SPIDER CHARTS FROM BINARY PATIENT MATRIX

    # num_features = 8
    # num_patients = 75
    [num_patients, num_features] = info_matrix.shape
    N = num_features

    # indices_male = (np.random.rand(num_patients, )>=0.5).astype(bool)
    # indices_female = np.logical_not(indices_male)

    # categories = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    # info_matrix = (np.random.rand(num_patients, num_features)>=0.5).astype(int)
    info_matrix_male = info_matrix[indices_male, :]
    info_matrix_female = info_matrix[indices_female, :]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    # ax = plt.(111, polar=True)
    fig = plt.figure(figsize=(6.4, 3.25))
    ax0 = fig.add_subplot(131, polar=True)
    ax1 = fig.add_subplot(132, polar=True)
    ax2 = fig.add_subplot(133, polar=True)

    # fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(12, 6), polar=True)

    # If you want the first axis to be on top:
    ax0.set_theta_offset(pi / 2)
    ax0.set_theta_direction(-1)
    ax1.set_theta_offset(pi / 2)
    ax1.set_theta_direction(-1)
    ax2.set_theta_offset(pi / 2)
    ax2.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    # plt.xticks(angles[:-1], categories)
    plt.sca(ax0)
    plt.xticks(angles[:-1], categories)
    plt.sca(ax1)
    plt.xticks(angles[:-1], categories)
    plt.sca(ax2)
    plt.xticks(angles[:-1], categories)

    aux_theta_shift = 360/18

    # Draw ylabels
    plt.sca(ax0)
    ax0.set_rlabel_position(aux_theta_shift)
    ax0.tick_params(axis='x', which='major', pad=15)
    plt.yticks(y_pos, y_pos_str, color="grey", size=7)
    plt.ylim(0, ylim_max)
    plt.sca(ax1)
    ax1.set_rlabel_position(aux_theta_shift)
    ax1.tick_params(axis='x', which='major', pad=15)
    plt.yticks(y_pos, y_pos_str, color="grey", size=7)
    plt.ylim(0, ylim_max)
    plt.sca(ax2)
    ax2.set_rlabel_position(aux_theta_shift)
    ax2.tick_params(axis='x', which='major', pad=15)
    plt.yticks(y_pos, y_pos_str, color="grey", size=7)
    plt.ylim(0, ylim_max)

    # Sum
    sum_value = np.sum(info_matrix, axis=0)
    sum_value_male = np.sum(info_matrix_male, axis=0)
    sum_value_female = np.sum(info_matrix_female, axis=0)
    # Percentage
    mean_value = sum_value/np.sum(info_matrix) * 100
    mean_value_male = sum_value_male/np.sum(info_matrix_male) * 100
    mean_value_female = sum_value_female/np.sum(info_matrix_female) * 100

    if percentage:
        values = mean_value.tolist()
        values_male = mean_value_male.tolist()
        values_female = mean_value_female.tolist()
    else:
        values = sum_value.tolist()
        values_male = sum_value_male.tolist()
        values_female = sum_value_female.tolist()

    values += values[:1]
    values_male += values_male[:1]
    values_female += values_female[:1]

    # All
    aux_label = 'All'
    ax0.plot(angles, values, linewidth=1, linestyle='solid', label=aux_label)
    ax0.fill(angles, values, 'r', alpha=0.1)
    # Male
    aux_label = 'Male'
    ax1.plot(angles, values_male, linewidth=1, linestyle='solid', label=aux_label)
    ax1.fill(angles, values_male, 'g', alpha=0.1)
    # Female
    aux_label = 'Female'
    ax2.plot(angles, values_female, linewidth=1, linestyle='solid', label=aux_label)
    ax2.fill(angles, values_female, 'b', alpha=0.1)

    # Add legend
    plt.sca(ax0)
    plt.title('All Patients')
    plt.sca(ax1)
    plt.title('Male')
    plt.sca(ax2)
    plt.title('Female')

    fig.suptitle(figure_title)

    plt.subplots_adjust(wspace=0.4)

    plt.show()


def plotCartoFinderPointsOnMeshCircles(VerticesA, FacesA, CartoFinderPoints, indices_all=[],
                                       indices_male = [], indices_female = [],
                                       figure_title='', CartoFinderPoints_color='blue', intensity=[0, 0.33, 0.66, 1], circle_radius=10,
                                       save_figure=False, save_figure_path='',
                                       show_grid=True, camera_settings='',
                                       opacity=1):

    if CartoFinderPoints.ndim == 1:
        CartoFinderPoints = np.reshape(CartoFinderPoints, (1,3))

    fig = go.Figure(layout_title_text=figure_title)

    plot_intensity = intensity.copy()
    num_cf_points = CartoFinderPoints.shape[0]
    noditos = []
    # circle_radius = 1
    for c in range(num_cf_points):
        aux_index = indices_all[c]
        aux_pos = CartoFinderPoints[c,:]
        # Find all nodes dist(node, point) < circle_radius
        aux_dist = (VerticesA - aux_pos) ** 2
        aux_dist = np.sum(aux_dist, axis=1)
        aux_dist = np.sqrt(aux_dist)
        aux_node_indices = np.where(aux_dist<=circle_radius)[0]
        noditos = np.append(noditos, aux_node_indices)

        if aux_index in indices_male:
            plot_intensity[aux_node_indices.astype(int)] = 1

        if aux_index in indices_female:
            plot_intensity[aux_node_indices.astype(int)] = 2
        # Male
        # Female
        # Find vertices containing the nodes
        # num_found_nodes = len(aux_node_indices)
        # aux_v = []
        # for n, n_index in enumerate(aux_node_indices):
        #     aux_v0_indices = np.where(FacesA[:, 0] == n_index)[0]
        #     aux_v1_indices = np.where(FacesA[:, 1] == n_index)[0]
        #     aux_v2_indices = np.where(FacesA[:, 2] == n_index)[0]
        #     aux_v = np.append(aux_v, aux_v0_indices)
        #     aux_v = np.append(aux_v, aux_v1_indices)
        #     aux_v = np.append(aux_v, aux_v2_indices)
        #
        # aux_v = np.unique(aux_v)
        # plot_intensity[aux_v.astype(int), 0] = 0

    # plot_intensity[noditos.astype(int), 0] = 0

    trace_meshA = go.Mesh3d(
        x=VerticesA[:, 0],
        y=VerticesA[:, 1],
        z=VerticesA[:, 2],
        # colorbar_title='z',
        colorscale=[[0, 'gold'],
                    [0.5, 'mediumturquoise'],
                    [1, 'magenta']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=plot_intensity,
        opacity=opacity,
        # i, j and k give the vertices of triangles
        # here we represent the 4 triangles of the tetrahedron surface
        i=FacesA[:, 0],
        j=FacesA[:, 1],
        k=FacesA[:, 2],
        name='meshA',
        showscale=True
    )

    # trace_A = go.Scatter3d(x=CartoFinderPoints[:,0],
    #                        y=CartoFinderPoints[:,1],
    #                        z=CartoFinderPoints[:,2],
    #                        name='MeshA points',
    #                        mode='markers', marker=dict(
    #         size=10,
    #         color=CartoFinderPoints_color,  # set color to an array/list of desired values
    #         colorscale='Viridis',  # choose a colorscale
    #         opacity=0.8
    #     ))

    fig.add_trace(trace_meshA)

    if show_grid == False:
        fig.update_layout(scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white", ),
            yaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white"),
            zaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white", ), )
        )
    # fig.add_trace(trace_A)
    #
    # noditos = np.unique(noditos)
    # noditos_coord = VerticesA[noditos.astype(int),:]
    # trace_noditos = go.Scatter3d(x=noditos_coord[:, 0],
    #                        y=noditos_coord[:, 1],
    #                        z=noditos_coord[:, 2],
    #                        mode='markers', marker=dict(
    #         size=10,
    #         color='red',  # set color to an array/list of desired values
    #         colorscale='Viridis',  # choose a colorscale
    #         opacity=0.8
    #     ))
    # fig.add_trace(trace_noditos)

    if camera_settings != '':


        fig.update_layout(scene_camera=camera_settings)

    fig.show()


def plotBarChart(info_matrix, indices_male, indices_female, categories, figure_title='', percentage=False,
                 save_figure_path='', show_pval=True, aux_y_offset=0.05, group_names=['Male', 'Female'], pval_th=0.05,
                 text_fontsize=20, pval_fontsize=16, xlegend=0.95, show_title=False, y_middle_label=''):
    ## CREATE SPIDER CHARTS FROM BINARY PATIENT MATRIX

    # num_features = 8
    # num_patients = 75
    [num_patients, num_features] = info_matrix.shape
    N = num_features

    info_matrix_male = info_matrix[indices_male, :]
    info_matrix_female = info_matrix[indices_female, :]

    num_males = len(indices_male)
    num_females = len(indices_female)

    num_patients = num_males + num_females

    # Sum
    sum_value = np.sum(info_matrix, axis=0)
    sum_value_male = np.sum(info_matrix_male, axis=0)
    sum_value_female = np.sum(info_matrix_female, axis=0)
    # Percentage
    # mean_value = sum_value/np.sum(info_matrix) * 100
    # mean_value_male = sum_value_male/np.sum(info_matrix_male) * 100
    # mean_value_female = sum_value_female/np.sum(info_matrix_female) * 100
    mean_value = sum_value/num_patients * 100
    mean_value_male = sum_value_male/num_males * 100
    mean_value_female = sum_value_female/num_females * 100

    if percentage:
        values = mean_value.tolist()
        values_male = mean_value_male.tolist()
        values_female = mean_value_female.tolist()
        ylabel = 'Percentage of patients with ' + y_middle_label + ' events (%)'
    else:
        values = sum_value.tolist()
        values_male = sum_value_male.tolist()
        values_female = sum_value_female.tolist()
        ylabel = 'Number of patients with ' + y_middle_label + ' events'

    # p-value
    proportional_pvals = np.zeros((N,))
    for a in range(N):
        aux_male_count = sum_value_male[a]
        aux_female_count = sum_value_female[a]
        aux_count = np.array([aux_male_count, aux_female_count])
        aux_nobs = np.array([num_males, num_females])
        aux_stat, aux_pval = proportions_ztest(aux_count, aux_nobs)
        proportional_pvals[a] = aux_pval

    aux_print = 'PLOT - Proportional p-vals: ' + str(proportional_pvals)
    print(aux_print)
    # >>> import numpy as np
    # >>> from statsmodels.stats.proportion import proportions_ztest
    # >>> count = np.array([5, 12])
    # >>> nobs = np.array([83, 99])
    # >>> stat, pval = proportions_ztest(count, nobs)
    # >>> print('{0:0.3f}'.format(pval))

    fig = go.Figure()

    # fig.add_trace(go.Bar(
    #     x=categories,
    #     y=values,
    #     name='All Patients'
    #     # marker_color='indianred'
    # ))
    fig.add_trace(go.Bar(
        x=categories,
        y=values_male,
        name=group_names[0]
        # marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=categories,
        y=values_female,
        name=group_names[1]
        # marker_color='lightsalmon'
    ))

    if show_pval:
        for a in range(N):
            aux_pval = proportional_pvals[a]

            # plot
            aux_str = '<0.001'
            if aux_pval > 0.001:
                # aux_str = str(aux_pval)
                aux_str = "{:.3f}".format(aux_pval)

            aux_area_name = categories[a]
            aux_y = np.max([values_male[a], values_female[a]]) + aux_y_offset

            if aux_pval < pval_th:
                # # Central line
                # fig.add_shape(type='line', x0=a, y0=aux_y, x1=a + 0.3, y1=aux_y,
                #               line=dict(color='Black', ), xref='x', yref='y')
                # # left vertical line
                # fig.add_shape(type='line', x0=a, y0=aux_y - aux_y_offset, x1=a, y1=aux_y,
                #               line=dict(color='Black', ), xref='x', yref='y')
                # # right vertical line
                # fig.add_shape(type='line', x0=a + 0.3, y0=aux_y - aux_y_offset, x1=a + 0.3, y1=aux_y,
                #               line=dict(color='Black', ), xref='x', yref='y')
                # # p-value text
                # fig.add_annotation(x=a + 0.15, y=0.5 + aux_y + aux_y_offset, text=aux_str, showarrow=False, font=dict(size=pval_fontsize))
                # Central line
                fig.add_shape(type='line', x0=a - 0.25, y0=aux_y, x1=a + 0.25, y1=aux_y,
                              line=dict(color='Black', ), xref='x', yref='y')
                # left vertical line
                fig.add_shape(type='line', x0=a - 0.25, y0=aux_y - aux_y_offset, x1=a - 0.25, y1=aux_y,
                              line=dict(color='Black', ), xref='x', yref='y')
                # right vertical line
                fig.add_shape(type='line', x0=a + 0.25, y0=aux_y - aux_y_offset, x1=a + 0.25, y1=aux_y,
                              line=dict(color='Black', ), xref='x', yref='y')
                # p-value text
                fig.add_annotation(x=a , y=0.5 + aux_y + aux_y_offset, text=aux_str, showarrow=False, font=dict(size=pval_fontsize))


    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    if show_title:
        aux_title = figure_title
    else:
        aux_title = ''
    fig.update_layout(barmode='group', xaxis_tickangle=-45, title_text=aux_title, title_x=0.5,
                      yaxis_title=ylabel, font=dict(size=text_fontsize),
                      legend=dict(yanchor="top",y=0.99,xanchor="right",x=xlegend)
                      )
    fig.show()

    if save_figure_path != '':
        fig.write_html(save_figure_path)
        # fig.write_image(save_figure_path + ".eps")
        # pio.write_image(fig, 'save_figure_path' + '.eps', width=700, height=775)
        # pio.write_image(fig, 'save_figure_path' + '.png', width=700, height=775)



def plotBarChartEventsInformation(info_matrix, indices_male, indices_female, categories, figure_title='', save_figure_path='', ylim='', ylabel='',
                                  show_pval=True, aux_y_offset=0.05, group_names=['Male', 'Female'], pval_th=0.05,
                                  text_fontsize=20, pval_fontsize=16, xlegend=0.95, show_title=False):
    ## CREATE SPIDER CHARTS FROM BINARY PATIENT MATRIX

    # num_features = 8
    # num_patients = 75
    [num_patients, num_features] = info_matrix.shape
    N = num_features

    info_matrix_male = info_matrix[indices_male, :]
    info_matrix_female = info_matrix[indices_female, :]

    info_matrix_binary = np.zeros(info_matrix.shape)
    for n in range(num_patients):
        for f in range(num_features):
            aux_val = info_matrix[n, f]

            if aux_val > 0:
                info_matrix_binary[n, f] = 1

    num_males = len(indices_male)
    num_females = len(indices_female)

    num_patients_with_events_per_area = np.sum(info_matrix_binary, axis=0)
    num_patients_with_events_per_area_male = np.sum(info_matrix_binary[indices_male, :], axis=0)
    num_patients_with_events_per_area_female = np.sum(info_matrix_binary[indices_female, :], axis=0)

    # Sum
    sum_value = np.sum(info_matrix, axis=0)
    sum_value_male = np.sum(info_matrix_male, axis=0)
    sum_value_female = np.sum(info_matrix_female, axis=0)

    # Mean value (wrt non-zero entries)
    mean_value = sum_value/num_patients_with_events_per_area
    mean_value_male = sum_value_male/num_patients_with_events_per_area_male
    mean_value_female = sum_value_female/num_patients_with_events_per_area_female
    # Correct nan values
    mean_value[np.isnan(mean_value)] = 0
    mean_value_male[np.isnan(mean_value_male)] = 0
    mean_value_female[np.isnan(mean_value_female)] = 0

    values = mean_value.tolist()
    values_male = mean_value_male.tolist()
    values_female = mean_value_female.tolist()
    # ylabel = 'Mean values of all the events in patients (ms)'

    fig = go.Figure()

    # fig.add_trace(go.Bar(
    #     x=categories,
    #     y=values,
    #     name='All Patients'
    #     # marker_color='indianred'
    # ))
    fig.add_trace(go.Bar(
        x=categories,
        y=values_male,
        name=group_names[0]
        # marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=categories,
        y=values_female,
        name=group_names[1]
        # marker_color='lightsalmon'
    ))

    if show_pval:
        pval_feature = np.zeros((num_features, 1))
        for a in range(num_features):
            aux_feature_male = info_matrix_male[:, a]
            aux_feature_male = aux_feature_male[np.where(aux_feature_male>0)[0]]
            aux_feature_female = info_matrix_female[:, a]
            aux_feature_female = aux_feature_female[np.where(aux_feature_female > 0)[0]]

            # Welch's t-test
            aux_a = np.squeeze(aux_feature_male)
            aux_b = np.squeeze(aux_feature_female)
            # print(aux_a, aux_b)
            [aux_ttest_val, aux_pval] = stats.ttest_ind(a=aux_a,
                                                        b=aux_b, equal_var=False)
            # print(aux_ttest_val, aux_pval)
            pval_feature[a, 0] = aux_pval

            # plot
            aux_str = '<0.001'
            if aux_pval > 0.001:
                # aux_str = str(aux_pval)
                aux_str = "{:.3f}".format(aux_pval)

            # aux_y_offset = 5

            aux_area_name = categories[a]
            aux_y = np.max([values_male[a], values_female[a]]) + aux_y_offset

            if aux_pval < pval_th:
                # Central line
                fig.add_shape(type='line', x0=a - 0.25, y0=aux_y, x1=a + 0.25, y1=aux_y,
                              line=dict(color='Black', ), xref='x', yref='y')
                # left vertical line
                fig.add_shape(type='line', x0=a - 0.25, y0=aux_y - aux_y_offset, x1=a - 0.25, y1=aux_y,
                              line=dict(color='Black', ), xref='x', yref='y')
                # right vertical line
                fig.add_shape(type='line', x0=a + 0.25, y0=aux_y - aux_y_offset, x1=a + 0.25, y1=aux_y,
                              line=dict(color='Black', ), xref='x', yref='y')
                # p-value text
                fig.add_annotation(x=a, y=aux_y + 0.5 + aux_y_offset, text=aux_str, showarrow=False, font=dict(size=pval_fontsize))


    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    if show_title:
        aux_title = figure_title
    else:
        aux_title = ''
    fig.update_layout(barmode='group', xaxis_tickangle=-45, title_text=aux_title, title_x=0.5,
                      yaxis_title=ylabel, font=dict(size=text_fontsize),
                      legend=dict(yanchor="top", y=0.99, xanchor="right", x=xlegend)
                      )

    if ylim != '':
        fig.update_layout(yaxis=dict(range=[ylim[0], ylim[1]]))

    fig.show()

    aux_print = 'P-VALUES - ' + figure_title + ' - ' + str(pval_feature)
    print(aux_print)

    if save_figure_path != '':
        fig.write_html(save_figure_path + '.html')
        # fig.write_image(save_figure_path + ".eps")

def plotCummulativeBarChart(info_matrix, indices_male, indices_female, categories, figure_title='', percentage=False,
                 save_figure_path='', show_pval=True, aux_y_offset=0.05, group_names=['Male', 'Female'], pval_th=0.05,
                 text_fontsize=20, pval_fontsize=16, xlegend=0.95, show_title=False, y_middle_label=''):

    # plotCummulativeBarChart(Cummulative_rotor_areas_binary[:aux_gender_len, :], aux_male_indices, aux_female_indices,
    #                         Cummulative_names, aux_title, percentage=True,
    #                         save_figure_path=aux_cummulative_rotor_bar_figure_path,
    #                         show_pval=True, aux_y_offset=0.5,
    #                         group_names=group_names, pval_th=0.05,
    #                         text_fontsize=22, pval_fontsize=18,
    #                         xlegend=0.99, show_title=False,
    #                         y_middle_label='rotor')


    ## CREATE SPIDER CHARTS FROM BINARY PATIENT MATRIX

    [num_patients, num_features] = info_matrix.shape
    N = num_features

    info_matrix_male = info_matrix[indices_male, :]
    info_matrix_female = info_matrix[indices_female, :]

    num_males = len(indices_male)
    num_females = len(indices_female)

    num_patients = num_males + num_females

    # Sum
    sum_value = np.sum(info_matrix, axis=0)
    sum_value_male = np.sum(info_matrix_male, axis=0)
    sum_value_female = np.sum(info_matrix_female, axis=0)
    # Percentage
    # mean_value = sum_value/np.sum(info_matrix) * 100
    # mean_value_male = sum_value_male/np.sum(info_matrix_male) * 100
    # mean_value_female = sum_value_female/np.sum(info_matrix_female) * 100
    mean_value = sum_value/num_patients * 100
    mean_value_male = sum_value_male/num_males * 100
    mean_value_female = sum_value_female/num_females * 100

    if percentage:
        values = mean_value.tolist()
        values_male = mean_value_male.tolist()
        values_female = mean_value_female.tolist()
        ylabel = 'Percentage of patients with ' + y_middle_label + ' events (%)'
    else:
        values = sum_value.tolist()
        values_male = sum_value_male.tolist()
        values_female = sum_value_female.tolist()
        ylabel = 'Number of patients with ' + y_middle_label + ' events'

    # p-value
    proportional_pvals = np.zeros((N,))
    for a in range(N):
        aux_male_count = sum_value_male[a]
        aux_female_count = sum_value_female[a]
        aux_count = np.array([aux_male_count, aux_female_count])
        aux_nobs = np.array([num_males, num_females])
        aux_stat, aux_pval = proportions_ztest(aux_count, aux_nobs)
        proportional_pvals[a] = aux_pval

    aux_print = 'PLOT - Proportional p-vals: ' + str(proportional_pvals)
    print(aux_print)

    fig = go.Figure()

    for n in range(N):
        aux_bar_name = categories[n]
        aux_y = info_matrix[:, n]
        # x male/female
        # name areas
        # fig.add_trace(go.Bar(x=group_names, y=aux_y, name=aux_bar_name))
        # # males
        # fig.add_trace(go.Bar(x=[group_names[0]], y=[values_male[n]], name=aux_bar_name))
        # # females
        # fig.add_trace(go.Bar(x=[group_names[1]], y=[values_female[n]], name=aux_bar_name))

        # both
        fig.add_trace(go.Bar(x=group_names, y=[values_male[n], values_female[n]], name=aux_bar_name))

    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    fig.show()

    # x = ['b', 'a', 'c', 'd']
    # fig = go.Figure(go.Bar(x=x, y=[2, 5, 1, 9], name='PVs'))
    # fig.add_trace(go.Bar(x=x, y=[1, 4, 9, 16], name='PVs + PV Antra'))
    # fig.add_trace(go.Bar(x=x, y=[6, 8, 4.5, 8], name='PVs + PV Antra + Posterior'))
    # fig.add_trace(go.Bar(x=x, y=[6, 8, 4.5, 8], name='Rest of areas'))
    #
    # fig.update_layout( xaxis={'categoryorder': 'total descending'})
    # fig.show()


    # fig.add_trace(go.Bar(
    #     x=categories,
    #     y=values_male,
    #     name=group_names[0]
    #     # marker_color='indianred'
    # ))
    # fig.add_trace(go.Bar(
    #     x=categories,
    #     y=values_female,
    #     name=group_names[1]
    #     # marker_color='lightsalmon'
    # ))
    #
    # if show_pval:
    #     for a in range(N):
    #         aux_pval = proportional_pvals[a]
    #
    #         # plot
    #         aux_str = '<0.001'
    #         if aux_pval > 0.001:
    #             # aux_str = str(aux_pval)
    #             aux_str = "{:.3f}".format(aux_pval)
    #
    #         aux_area_name = categories[a]
    #         aux_y = np.max([values_male[a], values_female[a]]) + aux_y_offset
    #
    #         if aux_pval < pval_th:
    #             # # Central line
    #             # fig.add_shape(type='line', x0=a, y0=aux_y, x1=a + 0.3, y1=aux_y,
    #             #               line=dict(color='Black', ), xref='x', yref='y')
    #             # # left vertical line
    #             # fig.add_shape(type='line', x0=a, y0=aux_y - aux_y_offset, x1=a, y1=aux_y,
    #             #               line=dict(color='Black', ), xref='x', yref='y')
    #             # # right vertical line
    #             # fig.add_shape(type='line', x0=a + 0.3, y0=aux_y - aux_y_offset, x1=a + 0.3, y1=aux_y,
    #             #               line=dict(color='Black', ), xref='x', yref='y')
    #             # # p-value text
    #             # fig.add_annotation(x=a + 0.15, y=0.5 + aux_y + aux_y_offset, text=aux_str, showarrow=False, font=dict(size=pval_fontsize))
    #             # Central line
    #             fig.add_shape(type='line', x0=a - 0.25, y0=aux_y, x1=a + 0.25, y1=aux_y,
    #                           line=dict(color='Black', ), xref='x', yref='y')
    #             # left vertical line
    #             fig.add_shape(type='line', x0=a - 0.25, y0=aux_y - aux_y_offset, x1=a - 0.25, y1=aux_y,
    #                           line=dict(color='Black', ), xref='x', yref='y')
    #             # right vertical line
    #             fig.add_shape(type='line', x0=a + 0.25, y0=aux_y - aux_y_offset, x1=a + 0.25, y1=aux_y,
    #                           line=dict(color='Black', ), xref='x', yref='y')
    #             # p-value text
    #             fig.add_annotation(x=a + 0.15, y=0.5 + aux_y + aux_y_offset, text=aux_str, showarrow=False, font=dict(size=pval_fontsize))
    #
    #
    # # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    if show_title:
        aux_title = figure_title
    else:
        aux_title = ''
    fig.update_layout(barmode='group', xaxis_tickangle=-45, title_text=aux_title, title_x=0.5,
                      yaxis_title=ylabel, font=dict(size=text_fontsize),
                      legend=dict(yanchor="top",y=0.99,xanchor="right",x=xlegend)
                      )
    fig.show()
    #
    # if save_figure_path != '':
    #     fig.write_html(save_figure_path + '.html')
    #     # fig.write_image(save_figure_path + ".eps")
    #     # pio.write_image(fig, 'save_figure_path' + '.eps', width=700, height=775)
    #     # pio.write_image(fig, 'save_figure_path' + '.png', width=700, height=775)


def plotCartoFinderPointsOnMeshCirclesVoltage(VerticesA, FacesA, CartoFinderPoints, indices_all=[],
                                       indices_male = [], indices_female = [],
                                       figure_title='', CartoFinderPoints_color='blue', intensity=[0, 0.33, 0.66, 1], circle_radius=10,
                                       save_figure=False, save_figure_path='',
                                       show_grid=True, camera_settings='', max_value=10):

    if CartoFinderPoints.ndim == 1:
        CartoFinderPoints = np.reshape(CartoFinderPoints, (1,3))

    fig = go.Figure(layout_title_text=figure_title)

    plot_intensity = intensity.copy()
    num_cf_points = CartoFinderPoints.shape[0]
    noditos = []
    # circle_radius = 1
    for c in range(num_cf_points):
        aux_index = indices_all[c]
        aux_pos = CartoFinderPoints[c,:]
        # Find all nodes dist(node, point) < circle_radius
        aux_dist = (VerticesA - aux_pos) ** 2
        aux_dist = np.sum(aux_dist, axis=1)
        aux_dist = np.sqrt(aux_dist)
        aux_node_indices = np.where(aux_dist<=circle_radius)[0]
        noditos = np.append(noditos, aux_node_indices)

        if aux_index in indices_male:
            plot_intensity[aux_node_indices.astype(int)] = max_value

        if aux_index in indices_female:
            plot_intensity[aux_node_indices.astype(int)] = max_value

    trace_meshA = go.Mesh3d(
        x=VerticesA[:, 0],
        y=VerticesA[:, 1],
        z=VerticesA[:, 2],
        # colorbar_title='z',
        colorscale=[[0, 'gold'],
                    [0.5, 'mediumturquoise'],
                    [0.999, 'magenta'],
                    [1, 'black']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=plot_intensity,
        # i, j and k give the vertices of triangles
        # here we represent the 4 triangles of the tetrahedron surface
        i=FacesA[:, 0],
        j=FacesA[:, 1],
        k=FacesA[:, 2],
        name='meshA',
        showscale=True
    )
    fig.add_trace(trace_meshA)

    if show_grid == False:
        fig.update_layout(scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white", ),
            yaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white"),
            zaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white", ), )
        )

    if camera_settings != '':
        fig.update_layout(scene_camera=camera_settings)

    fig.show()


def plotAblationOnMesh(VerticesA, FacesA, AblationPoints, figure_title='', AblationPoints_color='red',
                                intensity=[0, 0.33, 0.66, 1], show_grid=True, opacity=1,
                                legend_title_mesh='', legend_title_scatter='', clim_scatter=[0, 1], scatter_size=100,
                                colorscale_mesh='rainbow', invert_colorbar_mesh=False,
                                colorscale_scatter='RdBu', invert_colorbar_scatter=False,
                                mesh_name='', scatter_name='',
                                hover_info=[], hover_txt_str='',
                                save_figure=False, save_figure_path=''):
    if invert_colorbar_mesh:
        colorscale_mesh = colorscale_mesh + '_r'
    if invert_colorbar_scatter:
        colorscale_scatter = colorscale_scatter + '_r'

    if AblationPoints.ndim == 1:
        CartoFinderPoints = np.reshape(AblationPoints, (1,3))

    fig = go.Figure(layout_title_text=figure_title)

    trace_meshA = go.Mesh3d(
        x=VerticesA[:, 0],
        y=VerticesA[:, 1],
        z=VerticesA[:, 2],
        showlegend=True,
        hoverinfo='text',
        # colorbar_title='z',
        colorscale=colorscale_mesh,
        reversescale=True,
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=intensity,
        opacity=opacity,
        # i, j and k give the vertices of triangles
        # here we represent the 4 triangles of the tetrahedron surface
        i=FacesA[:, 0],
        j=FacesA[:, 1],
        k=FacesA[:, 2],
        name=mesh_name,
        showscale=True,
        colorbar=dict(title=legend_title_mesh, thicknessmode="pixels", thickness=15,
                      x=1, titleside='right')
    )

    aux_num_points = AblationPoints.shape[0]
    if len(hover_info)>0:
        aux_hovertext = [(hover_txt_str + ' ' + str(int(aux_val)) + ', ' + legend_title_scatter + ': ' + "{:.2f}".format(AblationPoints_color[a])) for a, aux_val in enumerate(hover_info)]
    else:
        aux_hovertext = [('P: [' + "{:.2f}".format(AblationPoints[a,0]) + ', ' + "{:.2f}".format(AblationPoints[a,1]) + ', ' + "{:.2f}".format(AblationPoints[a,2]) + ']') for a in range(aux_num_points)]

    trace_A = go.Scatter3d(x=AblationPoints[:,0],
                           y=AblationPoints[:,1],
                           z=AblationPoints[:,2],
                           hovertext=aux_hovertext,
                           showlegend=True,
                           name=scatter_name,
                           mode='markers', marker=dict(
            size=scatter_size,
            color=AblationPoints_color,  # set color to an array/list of desired values
            colorscale=colorscale_scatter,  # choose a colorscale
            opacity=0.8,
            cmin=clim_scatter[0],
            cmax=clim_scatter[1],
            colorbar=dict(title=legend_title_scatter, thicknessmode="pixels", thickness=15,
                          x=1.05, titleside='right')
        ))

    fig.add_trace(trace_meshA)
    fig.add_trace(trace_A)
    trace_meshA.opacity = 0.1
    fig.add_trace(trace_meshA)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    if show_grid == False:
        fig.update_layout(scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white", ),
            yaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white"),
            zaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=False,
                zerolinecolor="white", ), )
        )
    fig.update_layout(title_text=figure_title, title_x=0.5)
    fig.show()

    if save_figure and save_figure_path != '':
        fig.write_html(save_figure_path)

