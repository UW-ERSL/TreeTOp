"""Linear Structural Finite Element Analysis."""

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

import mesher
import material
import bound_cond


_BilinMesh = mesher.BilinearStructMesher


class FEA:
  def __init__(self,
               mesh: _BilinMesh,
               mat_const: material.Material,
               bc: bound_cond.BC):
    self.mesh, self.mat_const, self.bc = mesh, mat_const, bc
    self.D0 = self.FE_compute_element_stiffness()


  def FE_compute_element_stiffness(self) -> np.ndarray:
    E = 1.
    nu = self.mat_const.poissons_ratio
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,\
                -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    D0 = E/(1-nu**2)*np.array([\
    [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]).T
    return D0


  def compute_elem_stiffness_matrix(self,
                                    youngs_modulus: jnp.ndarray
                                    )->jnp.ndarray:
    """
    Args:
      youngs_modulus: Array of size (num_elems,) which contain the modulus
        of each element
    Returns: Array of size (num_elems, 8, 8) which is the structual
      stiffness matrix of each of the bilinear quad elements. Each element has
      8 dofs corresponding to the x and y displacements of the 4 noded quad
      element.
    """
    # e - element, i - elem_nodes j - elem_nodes
    return jnp.einsum('e, ij->ije', youngs_modulus, self.D0)


  def assemble_stiffness_matrix(self,
                                elem_stiff_mtrx: jnp.ndarray
                                ) -> jnp.ndarray:
    """
    Args:
      elem_stiff_mtrx: Array of size (num_elems, 8, 8) which is the structual
        stiffness matrix of each of the bilinear quad elements. Each element has
        8 dofs corresponding to the x and y displacements of the 4 noded quad
        element.
    Returns: Array of size (num_dofs, num_dofs) which is the assembled global
      stiffness matrix.
    """
    glob_stiff_mtrx = jnp.zeros((self.mesh.num_dofs, self.mesh.num_dofs))
    glob_stiff_mtrx = glob_stiff_mtrx.at[(self.mesh.iK, self.mesh.jK)].add(
                                      elem_stiff_mtrx.flatten('F'))
    return glob_stiff_mtrx


  def solve(self, glob_stiff_mtrx)->jnp.ndarray:
    """Solve the system of Finite element equations.
    Args:
      glob_stiff_mtrx: Array of size (num_dofs, num_dofs) which is the assembled
        global stiffness matrix.
    Returns: Array of size (num_dofs,) which is the displacement of the nodes.
    """
    k_free = glob_stiff_mtrx[self.bc.free_dofs,:][:,self.bc.free_dofs]

    u_free = jax.scipy.linalg.solve(
          k_free,
          self.bc.force[self.bc.free_dofs],
          assume_a = 'pos', check_finite=False)
    u = jnp.zeros((self.mesh.num_dofs))
    u = u.at[self.bc.free_dofs].add(u_free.reshape(-1))
    return u


  def compute_compliance(self, u:jnp.ndarray)->jnp.ndarray:
    """Objective measure for structural performance.
    Args:
      u: Array of size (num_dofs,) which is the displacement of the nodes
        of the mesh.
    Returns: Structural compliance, which is a measure of performance. Lower
      compliance means stiffer and stronger design.
    """
    return jnp.dot(u, self.bc.force.flatten())


  def loss_function(self, density:jnp.ndarray)->float:
    """Wrapper function that takes in density field and returns compliance.
    Args:
      density: Array of size (num_elems,) that contain the density of each
        of the elements for FEA.
    Returns: Structural compliance, which is a measure of performance. Lower
      compliance means stiffer and stronger design.
    """
    youngs_modulus = material.compute_simp_material_modulus(density, self.mat_const)
    elem_stiffness_mtrx = self.compute_elem_stiffness_matrix(youngs_modulus)
    glob_stiff_mtrx = self.assemble_stiffness_matrix(elem_stiffness_mtrx)
    u = self.solve(glob_stiff_mtrx)
    return self.compute_compliance(u)


  def plot_displacement(self, u, density = None):
    elemDisp = u[self.edofMat].reshape(self.mesh.nelx*self.mesh.nely, 8)
    elemU = (elemDisp[:,0] + elemDisp[:,2] + elemDisp[:,4] + elemDisp[:,6])/4
    elemV = (elemDisp[:,1] + elemDisp[:,3] + elemDisp[:,5] + elemDisp[:,7])/4
    delta = np.sqrt(elemU**2 + elemV**2)
    x, y = np.mgrid[:self.mesh.nelx, :self.mesh.nely]
    scale = 0.1*max(self.mesh.nelx, self.mesh.nely)/max(delta)
    x = x + scale*elemU.reshape(self.mesh.nelx, self.mesh.nely)
    y = y + scale*elemV.reshape(self.mesh.nelx, self.mesh.nely)

    if density is not None:
      delta = delta*np.round(density)
    z = delta.reshape(self.mesh.nelx, self.mesh.nely)
    im = plt.pcolormesh(x, y, z, cmap='coolwarm')
    plt.title('deformation')
    plt.colorbar(im)