"""Linear Structural FEA boundary conditions."""

import dataclasses
from enum import Enum, auto
import numpy as np
import jax.numpy as jnp

import mesher
_BilinMesh = mesher.BilinearStructMesher


@dataclasses.dataclass
class BC:
  """
  Attributes:
    force: Array of size (num_dofs, 1) that contain the imposed load on each dof.
    fixed_dofs: Array of size (num_fixed_dofs,) that contain all the dof numbers
      that are fixed.
  """
  force: jnp.ndarray
  fixed_dofs: np.ndarray

  @property
  def num_dofs(self):
    return self.force.shape[0]

  @property
  def free_dofs(self):
    return np.setdiff1d(np.arange(self.num_dofs), self.fixed_dofs)


class SturctBCs(Enum):
  MID_CANT_BEAM = auto()
  TIP_CANT_BEAM = auto()
  MBB_BEAM = auto()

def get_sample_struct_bc(mesh:_BilinMesh, sample:SturctBCs)->BC:
  """Get precoded sample structural boundary conditions."""

  force = np.zeros((mesh.num_dofs, 1))
  dofs=np.arange(mesh.num_dofs)

  if(sample == SturctBCs.MID_CANT_BEAM):
    fixed = dofs[0:2*(mesh.nely+1):1]
    force[mesh.num_dofs - 1*(mesh.nely+1), 0] = -1.

  elif(sample == SturctBCs.TIP_CANT_BEAM):
    fixed = dofs[0:2*(mesh.nely+1):1]
    force[mesh.num_dofs - 2*(mesh.nely+1), 0] = -1.

  elif(sample == SturctBCs.MBB_BEAM):
    fixed= np.union1d(np.arange(0,2*(mesh.nely+1),2),
                      mesh.num_dofs-2*(mesh.nely+1)+1)
    force[2*(mesh.nely+1)+1, 0]= -1.

  return BC(force=force, fixed_dofs=fixed)