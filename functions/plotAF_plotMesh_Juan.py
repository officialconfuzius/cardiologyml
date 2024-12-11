import plotly.graph_objs as go
import numpy as np
def plotMesh_Juan(vertices, triangles, triangles_color, clim_limits=[0, 0], intensity_mode='vertex',
                  figure_title='Mesh plot', representation='', legend_title='', color_scheme='Plasma',
                  fig='', save_images=False, save_figure_path='', row='', col='', scene_name='', show_grid=True,
                  atrial_areas = ['LAA', 'LSPV', 'LIPV', 'RIPV', 'RSPV', 'PW', 'AR', 'AW', 'AFL', 'AS', 'LW', 'MV']):
    # TO BE DONE
    num_triangles = triangles.shape[0]

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    aux_x_range = np.max(x) - np.min(x)
    aux_y_range = np.max(y) - np.min(y)
    aux_z_range = np.max(z) - np.min(z)
    aux_x_aspect = aux_x_range / aux_x_range
    aux_y_aspect = aux_y_range / aux_x_range
    aux_z_aspect = aux_z_range / aux_x_range

    aux_intensity = triangles_color
    # aux_intensity[aux_intensity>clim_limits[1]] = clim_limits[1]
    # aux_intensity[aux_intensity < clim_limits[0]] = clim_limits[0]

    if fig == '':
        fig = go.Figure(layout_title_text=figure_title)

    if (representation == '') | (representation == 'ent'):
        aux_hovertext = [str(f'{aux_val:.2f}') for aux_val in aux_intensity]
    elif representation == 'etp':
        # aux_hovertext = [(str(f'{aux_val*100:.2f}')+'%') for aux_val in aux_intensity]
        aux_hovertext = [(str(int(aux_val*100))+'%') for aux_val in aux_intensity]
    elif representation == 'groups':
        # atrial_areas = ['LAA', 'LSPV', 'LIPV', 'RIPV', 'RSPV', 'PW', 'AR', 'AW', 'AFL', 'AS', 'LW', 'MV']
        aux_hovertext = [('Group ' + str(int(aux_val)) + ': ' + atrial_areas[int(aux_val)]) for aux_val in aux_intensity]
    elif representation == 'groups_points':
        aux_hovertext = [('Group ' + str(int(aux_val)) + ': ' + atrial_areas[int(aux_val)]) + '\nP('+ str(int(a)) +
                         '): [' + "{:.2f}".format(x[a]) + ', ' + "{:.2f}".format(y[a]) + ', ' + "{:.2f}".format(z[a]) + ']' for a, aux_val in
                         enumerate(aux_intensity)]

    trace_mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        hovertext=aux_hovertext,
        hoverinfo='text',
        # color='blue',
        # colorbar_title='z',
        # colorscale=[[0, 'gold'],
        #             [0.5, 'mediumturquoise'],
        #             [1, 'magenta']],

        # Intensity of each vertex, which will be interpolated and color-coded
        # intensity=[0, 0.33, 0.66, 1],
        # if triangles_color.shape[0] == vertices.shape[0]:
        intensity=aux_intensity,
        intensitymode=intensity_mode,
        # elif triangles_color.shape[0] == triangles.shape[0]:
        # facecolor = aux_intensity,
        colorscale=color_scheme,
        cmin=clim_limits[0],
        cmax=clim_limits[1],
        # i, j and k give the vertices of triangles
        # here we represent the 4 triangles of the tetrahedron surface
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        name='mesh',
        showscale=True,
        colorbar=dict(title=legend_title, thicknessmode="pixels", thickness=5,
                      tickvals=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], x=1, titleside='right'))

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

    # fig.show(renderer='browser')

    if save_images & (save_figure_path != ''):
        fig.write_html(save_figure_path)

    return fig
