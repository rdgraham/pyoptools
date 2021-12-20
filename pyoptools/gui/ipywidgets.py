#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with functions and classes to represent the pyoptools objects
in `jupyter notebooks <http://jupyter.org>`_.
"""

from pyoptools.raytrace.system import System
from pyoptools.raytrace.component import Component
from pyoptools.raytrace.surface import Surface
from pyoptools.misc.pmisc import wavelength2RGB, cross, rot_x, rot_y, rot_z
try:
    import pythreejs as py3js
except ModuleNotFoundError:
    print("need py3js installed to be able to plot systems in Jupyter notebooks")
from pyoptools.misc.pmisc import wavelength2RGB, cross
from numpy import pi,array,dot
import numpy as np
from math import sqrt, exp
from matplotlib import colors
from matplotlib.cm import get_cmap

__all__ = ['Plot3D', 'PlotScene']

def surf2mesh(S,P=(0,0,0),D=(0,0,0),wire=False):

    color="#ffff00"

    points,polylist = S.polylist()

    #Conversion para quethreejs la entienda

    polylist=list(polylist)

    lpoly=[]
    lpoints=[]

    for l in points:
        lpoints.append(list(l))

    for l in polylist:
        lpoly.append(list(map(int,l)))

    vertices = lpoints

    faces = lpoly

    # Map the vertex colors into the 'color' slot of the faces
    # Map the normals
    nfaces=[]
    for f in faces:
        p0 = points[f[0]]
        p1 = points[f[1]]
        p2 = points[f[2]]
        v0 = array(p1)-array(p0)
        v1 = array(p2)-array(p0)
        v3 = cross(v0, v1)
        v3 = tuple(v3 / sqrt(v3[0]**2 + v3[1]**2 + v3[2]**2))

        nfaces.append(f + [v3, color, None])

    # Create the geometry:

    surfaceGeometry = py3js.Geometry(vertices=vertices,
        faces=nfaces,
        #colors=vertexcolors
                           )


    #surfaceGeometry = py3js.SphereGeometry(radius=300, widthSegments=32, heightSegments=24)

    if wire:
        surfaceGeometry = py3js.WireframeGeometry(surfaceGeometry)

    # Calculate normals per face, for nice crisp edges:
    surfaceGeometry.exec_three_obj_method('computeFaceNormals')

    surfaceMaterial=py3js.MeshPhongMaterial( color=color,
                                             ambient="#050505",
                                             specular="#ffffff",
                                             shininess= 15,
                                             emissive="#000000",
                                             side='DoubleSide',
                                             transparent = True,
                                             opacity=.8)
    #surfaceMaterial = py3js.MeshLambertMaterial(color='red',side='DoubleSide')

    # Create a mesh. Note that the material need to be told to use the vertex colors.
    surfaceMesh = py3js.Mesh(
        geometry=surfaceGeometry,
        material= surfaceMaterial,)

    surfaceMesh.rotation=*D,"ZYX"
    surfaceMesh.position=tuple(P)
    return surfaceMesh

def comp2mesh(C, P, D):
    c=py3js.Group()
    if isinstance(C, Component):
        for surf in C.surflist:
            sS, sP, sD = surf
            s=surf2mesh(sS,sP,sD)
            c.add(s)

    elif isinstance(C, System):
       for comp in C.complist:
            sC, sP, sD = comp
            c.add(comp2mesh(sC, sP, sD))
    #glPopMatrix()
    c.rotation=*D,"ZYX"
    c.position=tuple(P)
    return c


def ray2list(ray):
    rays=[]

    P1 = ray.pos
    if len(ray.childs) > 0:
        P2 = ray.childs[0].pos
    else:
        P2 = P1 + 10. * ray.dir

    if ray.intensity != 0:

        line=[list(P1),list(P2)]
        rays.append(line)

    for i in ray.childs:
        rays.extend(ray2list(i))
    return rays

def ray2mesh(ray):
    rays=py3js.Group()

    if ray.draw_color is None:
        color = wavelength2RGB(ray.wavelength)
    else:
        color = colors.to_rgb(ray.draw_color)

    int_colors = [int(255*c) for c in color]
    material = py3js.LineBasicMaterial(color = "#{:02X}{:02X}{:02X}".format(*int_colors))

    rl = ray2list(ray)

    for r in rl:
        geometry = py3js.Geometry()
        geometry.vertices =  r
        line = py3js.Line( geometry, material)
        rays.add(line)

    return rays


#def ray2mesh(ray):
#    rays=py3js.Group()

#    P1 = ray.pos
#    w = ray.wavelength
#    rc, gc, bc = wavelength2RGB(w)
#    rc=int(255*rc)
#    gc=int(255*gc)
#    bc=int(255*bc)
#    material = py3js.LineBasicMaterial(color = "#{:02X}{:02X}{:02X}".format(rc,gc,bc))

#    if len(ray.childs) > 0:
#        P2 = ray.childs[0].pos
#    else:
#        P2 = P1 + 10. * ray.dir

#    if ray.intensity != 0:

#        geometry = py3js.Geometry()

#        geometry.vertices =  [list(P1),list(P2)]

#        line = py3js.Line( geometry, material)

#        rays.add(line)

#    for i in ray.childs:
#        rays.add(ray2mesh(i))
#    return rays


def sys2mesh(os):
    s=py3js.Group()
    if os is not None:
        for i in os.prop_ray:
            s.add(ray2mesh(i))
        # Draw Components
        n=0
        for comp in os.complist:
            C, P, D = comp
            c=comp2mesh(C, P, D)
            s.add(c)
    return s

def Plot3D(S,size=(800,200),center=(0,0,0), rot=[(pi/3., pi/6., 0 )],
            scale=1, plot_scene=None):
    """Function to create 3D interactive visualization widgets in a jupyter
    notebook

    Args:
        S: (:class:`~pyoptools.raytrace.system.System`,
            :class:`~pyoptools.raytrace.component.Component` or
            :class:`~pyoptools.raytrace.component.Component`) Object to plot
        size: (Tuple(float,float)) Field of view in X and Y for the window
            shown in the notebook.
        center: (Tuple(float,float,float) Coordinate of the center of the
            visualization window given in the coordinate system of the object
            to plot.
        rot:   List of tuples. Each tuple describe an (Rx, Ry, Rz) rotation and
               are applied in order to generate the first view of the window.
        scale: (float)  Scale factor applied to the rendered window
        scene: PlotScene or None. A PlotScene object defining additional
               attributes of the rendered scene and cosmetic elements.
    Returns:
        pyjs renderer needed to show the image in the jupiter notebook.

    """
    width,height=size

    if plot_scene is None:
        plot_scene = PlotScene()

    light =  py3js.DirectionalLight(color='#ffffff',
                                    intensity=.7,
                                    position=[0, 1000,0])
    alight =  py3js.AmbientLight(color=plot_scene.ambient_color)


    # Set up a scene and render it:
    #cam = py3js.PerspectiveCamera(position=[0, 0, 500], fov=70, children=[light], aspect=width / height)

    pos = array((0, 0, 500))

    for r in rot:
        pos=dot(rot_z(r[2]),pos)
        pos=dot(rot_y(r[1]),pos)
        pos=dot(rot_x(r[0]),pos)

    cam = py3js.OrthographicCamera(-width/2*scale,width/2*scale, height/2*scale,
                                   -height/2*scale,children=[light],
                                   position=list(pos),
                                   zoom=scale)

    if isinstance(S,System):
        c=sys2mesh(S)
    elif isinstance(S,Component):
        c=comp2mesh(S,(0,0,0),(0,0,0))
    else:
        c=surf2mesh(S,(0,0,0),(0,0,0))

    plot_scene._set_bbox(c)

    children = [c, alight, cam]
    children.extend(plot_scene.get_all_elements())

    py3_scene = py3js.Scene(children=children,
                            background=plot_scene.background_color,
                            antialias=True)
    oc=py3js.OrbitControls(controlling=cam)
    oc.target=center
    #renderer = py3js.Renderer(camera=cam, background='black', background_opacity=1,
    renderer = py3js.Renderer(camera=cam,
                              scene=py3_scene,
                              controls=[oc],width=width*scale, height=height*scale)
    return(renderer)


def _gradient_texture(bg):
    """Returns a 2D gradient texture given a color map definition.
       Color map definition can be either a matplotlib.Colormap
       or a string with prefix 'gradient:' and suffix a valid matplotlib
       color map name.
    """
    height, width = (128,128)

    if isinstance(bg, colors.Colormap):
        cmap = bg
    elif bg.startswith('gradient:'):
        cmap_name = bg.split(':')[-1]
        cmap = get_cmap(cmap_name)
    else:
        raise ValueError(str(cmap_name)+' is an invalid background definition.')

    gradient = np.ones([height, width, 4], dtype=np.float32)
    for x, frac in enumerate(np.linspace(0, 1, num=height)):
        for y in range(width):
            gradient[x,y] = np.array(cmap(frac))

    texture = py3js.DataTexture(data=gradient, format='RGBAFormat', type='FloatType')
    return texture

class PlotScene(object):
    def __init__(self, background='gradient:bone',
                       ambient_color='#777777'):
        """ Constructs a plot scene object for 3D plotting which defines the
        plot background and can contain cosmetic elements such as grids and
        corners.

        Parameters
        ----------

        background : str or matplotlib.colors.Colormap
                     The background to use in the scene. Can be either a
                     matplotlib color definition (e.g. 'black' or '#000000'),
                     a string with the prefix 'gradient:' and suffix a valid
                     matplotlib colormap name, or a matplotlib colormap
                     can be passed directly. If a gradient is defined for the
                     background, a bounding sphere will be drawn around the
                     scene with the specified gradient as a texture.
        ambient_color : str
                     A matplolib color definition for the color of the ambient
                     light in the 3d scene.
        """

        self._gradient_background = None
        self.background_defn = background
        self.set_background()

        self.ambient_color = ambient_color
        self._grids = []
        self._bbox = None

    def _set_bbox(self, elements):
        self._bbox = (100, 100, 100)

    def get_all_elements(self):
        el = self._grids

        # make only at the at the last moment so that bounding box will be
        # correct with all the items in it
        if self.has_bounding_sphere() and self._bbox is not None:
            radius = max(self._bbox)
            print('Using maximum bounding radius ', radius)
            sphere = self._make_bounding_sphere(radius)
            if sphere is not None:
                el.append(sphere)

        return el

    @property
    def background_color(self):
        return self._background_color

    def set_background(self):
        if self.has_bounding_sphere():
            # Black background in this case, the background sphere with
            # gradient texture will only be drawn once all elements are
            # available so that they will all be enclosed
            self._background_color = None
        else:
            self._background_color = colors.to_hex(self.background_defn)


    def has_bounding_sphere(self):
        "Returns true if scene will have a bounding sphere drawn"
        b = self.background_defn
        return (isinstance(b, str) and b.startswith('gradient:')) or isinstance(b, colors.Colormap)

    def _make_bounding_sphere(self, max_radius):
        "Returns a textured bounding sphere for the scene"

        geometry = py3js.SphereGeometry(radius=max_radius,
                                        width_segments=128,
                                        height_segments=128)
        material = py3js.MeshLambertMaterial(color='white',
                                             map=_gradient_texture(self.background_defn),
                                             side='BackSide')
        sphere = py3js.Mesh(geometry=geometry, material=material)
        return sphere

    @property
    def ambient_color(self):
        return self._ambient_color

    @ambient_color.setter
    def ambient_color(self, color):
        self._ambient_color = colors.to_hex(color)


    def addGrid(self, axis='xz', size=10, divisions=10,
                      color_center_line='#444444', color_grid='#888888'):
        """Adds a cosmetic grid to the plot scene.

        Parameters
        ---------

        axis : str
               String description of the axes over which to draw the grid,
               can be withe 'xz', 'xy' or 'yz', or equivalently 'zx', 'yx' or
               'zy'.
        size : float
               length of the size of the grid (grid is always square) in mm.
        divisions : int
                  : number of divisions to use in the grid
        color_center_line : str
                            matplotlib compatible color descriptor for the
                            color of the center line of the grid
        color_center_grid : str
                            matplotlib compatible color descriptor for the
                            color of the grid
        """

        try:
            axis = axis.lower()
        except AttributeError:
            raise ValueError('Invalid axis descriptor for grid')

        grid = py3js.GridHelper(size=size, divisions=divisions,
                                colorCenterLine=colors.to_hex(color_center_line),
                                colorGrid=colors.to_hex(color_grid))

        if axis == 'xz' or axis == 'zx':
            pass
        elif axis == 'xy' or axis == 'yx':
            grid.rotateX( pi/2 )
        elif axis == 'yz' or axis == 'zy':
            grid.rotateX( pi/2 )
            grid.rotateZ( pi/2 )
        else:
            raise ValueError('Invalid axis descriptor for grid')

        self._grids.append(grid)
