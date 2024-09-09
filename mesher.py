import numpy as np
import dataclasses
from typing import Tuple

import utils

_Ext = utils.Extent

@dataclasses.dataclass
class BoundingBox:
  x: _Ext
  y: _Ext


class GridMesher:
  """
  Attributes:
    num_dim: number of dimensions of the mesh. Currently only handles 2D
    nelx: number of elements along X axis
    nely: number of elements along Y axis
    num_elems: number of elements in the mesh
    bounding_box: Contains the max and min coordinates of the mesh
    lx: length of domain along X axis
    ly: length of domain along Y axis
    elem_size: Array which contains the size of the element along X and Y axis
    elem_area: Area of each element
    domain_volume: Area of the rectangular domain
    num_nodes: number of nodes in the mesh. Assume a bilinear quad element
    elem_centers: Array of size (num_elems, 2) which are the coordinates of the
      centers of the element
  """

  @property
  def num_dim(self)->int:
    return 2


  def __init__(self, nelx:int, nely:int, bounding_box: BoundingBox):
    self.nelx, self.nely = nelx, nely
    self.num_elems = nelx*nely
    self.bounding_box = bounding_box
    self.lx = np.abs(self.bounding_box.x.range)
    self.ly = np.abs(self.bounding_box.y.range)
    dx, dy = self.lx/nelx, self.ly/nely
    self.elem_size = np.array([dx, dy])
    self.elem_area = dx*dy*np.ones((self.num_elems,))
    self.domain_volume = np.sum(self.elem_area)
    self.num_nodes = (nelx+1)*(nely+1)

    [x_grid, y_grid] = np.meshgrid(
               np.linspace(dx/2., self.lx-dx/2., nelx),
               np.linspace(dy/2., self.ly-dy/2., nely))
    [x_grid, y_grid] = np.meshgrid(
               np.linspace(self.bounding_box.x.min + dx/2., 
                           self.bounding_box.x.max-dx/2., 
                           nelx),
               np.linspace(self.bounding_box.y.min + dy/2.,
                           self.bounding_box.y.max-dy/2.,
                           nely))
    self.elem_centers = np.stack((x_grid, y_grid)).T.reshape(-1, self.num_dim)



class BilinearStructMesher(GridMesher):
  """Bilinear structured grid mesher for structural elasticity."""

  @property
  def nodes_per_elem(self)->int:
    return 4


  @property
  def dofs_per_node(self)->int:
    return 2


  def __init__(self, nelx:int, nely:int, bounding_box: BoundingBox):
    super().__init__(nelx, nely, bounding_box)
    self.num_nodes = (self.nelx + 1)*(self.nely + 1)
    self.dofs_per_elem = self.dofs_per_node*self.nodes_per_elem
    self.num_dofs = self.dofs_per_node*self.num_nodes
    (self.elem_node, self.edofMat,
      self.iK, self.jK) = self.compute_connectivity_info()


  def compute_connectivity_info(self)-> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    """Computes global and local mesh numbering.

    Returns:
      elem_node: Array of (nodes_per_elem, num_elems) that contain the global
        node numbers of the elements.
      edofMat: Array of (num_elems, dofs_per_elem) that contain the global
        dof numbers of the elements.
      iK: Array (num_elems * dofs_per_elem * dofs_per_elem) that contains the 
        global node number indices (row positions) corresponding to each 
        element's degrees of freedom within the assembled stiffness matrix.
      jK: Array (num_elems * dofs_per_elem * dofs_per_elem) that contains the 
        global node number indices (column positions) corresponding to each
        element's degrees of freedom within the assembled stiffness matrix.
    """
    elem_node = np.zeros((self.nodes_per_elem, self.num_elems))
    for elx in range(self.nelx):
      for ely in range(self.nely):
        el = ely+elx*self.nely
        n1=(self.nely+1)*elx+ely
        n2=(self.nely+1)*(elx+1)+ely
        elem_node[:,el] = np.array([n1+1, n2+1, n2, n1])
    elem_node = elem_node.astype(int)

    edofMat = np.zeros((self.num_elems,
        self.nodes_per_elem*self.num_dim), dtype=int)

    for elem in range(self.num_elems):
      enodes = elem_node[:, elem]
      edofs = np.stack((2*enodes, 2*enodes+1), axis=1).reshape(
                                            (1, self.dofs_per_elem))
      edofMat[elem, :] = edofs

    matrx_size = self.num_elems*self.dofs_per_elem**2
    iK = tuple(np.kron(edofMat, np.ones((self.dofs_per_elem, 1),dtype=int)).T
                .reshape(matrx_size, order ='F'))
    jK = tuple(np.kron(edofMat, np.ones((1, self.dofs_per_elem),dtype=int)).T
                .reshape(matrx_size, order ='F'))
    return elem_node, edofMat, iK, jK