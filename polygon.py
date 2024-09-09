import jax.numpy as jnp
import jax
import numpy as np
import dataclasses
from enum import Enum, auto
import mesher
import utils

_Ext = utils.Extent
_mesh = mesher.GridMesher

@dataclasses.dataclass
class PolygonExtents:
  """Hyper-parameters of the size, number and configurations of the polygons.
  Attributes:
    num_polys: number of polygons that occupy the design domain.
    num_planes_in_a_poly: number of planes that define each polygon.
    center_x: center coordinate extents of the polygons along x-axis.
    center_y: center coordinate extents of the polygons along y-axis.
    angle_offset: angular offset extents of the polygons.
    face_offset: face offset extents of the polygons from the center.
  """
  num_polys: int
  num_planes_in_a_poly: int
  center_x: _Ext
  center_y: _Ext
  angle_offset: _Ext
  face_offset: _Ext


@dataclasses.dataclass
class ConvexPolys:
  """A dataclass to describe convex polys.
  
  Each poly is described by a set of hyperplanes. The polys are inspired from:

  Deng, B., etal (2020). CVXNet: Learnable convex decomposition. CVPR, 31–41.
  https://doi.org/10.1109/CVPR42600.2020.00011

  However to ensure that the hyperplanes always form a bounded region the normals
  are fixed and is decided by the number of planes (N) in each poly. We assume the
  normal angle of the first plane is 0, (2pi)/N in the second, and i(2pi)/N of
  the ith plane. Each plane is further offset by a prescribed distance from the
  center of each poly.

  Attributes:
    center_x: Array of size (num_polys,) with the x-center of the polys.
    center_y: Array of size (num_polys,) with the y-center of the polys.
    angle_offset: Array of size (num_polys,) of the angle offset of the polys in
      radians.
    face_offset: Array of size (num_polys, num_planes_in_a_polys) with the 
      offset of each of the planes of the faces.
  """
  center_x: jnp.ndarray
  center_y: jnp.ndarray
  angle_offset: jnp.ndarray
  face_offset: jnp.ndarray

  def __post_init__(self):
    if(self.num_planes_in_a_poly < 3):
      raise Exception(f"Expect atleast 3 sides, got {self.num_planes_in_a_poly}")

  @property
  def num_free_parameters(self):
    """Number of free parameters for optimization."""
    return self.num_polys*(self.num_planes_in_a_poly + 3)

  @property
  def face_angles(self):
    """ Array of size (num_polys, num_planes_in_poly) """
    return (self.angle_offset[:, np.newaxis] + 
             jnp.linspace(0, 2*np.pi, self.num_planes_in_a_poly + 1)[:-1])

  @property
  def num_polys(self):
    return self.face_offset.shape[0]

  @property
  def num_planes_in_a_poly(self):
    return self.face_offset.shape[1]

  @classmethod
  def from_array(
      cls,
      state_array: jnp.ndarray,
      num_polys: int,
      num_planes_in_a_poly: int,
  ) -> 'ConvexPolys':
    """Converts a rank-1 array into `ConvexPolys`."""
    cx = state_array[0:num_polys]
    cy = state_array[num_polys:2*num_polys]
    ang = state_array[2*num_polys:3*num_polys]
    offset = state_array[3*num_polys:].reshape((num_polys, num_planes_in_a_poly))

    return ConvexPolys(cx, cy, ang, offset)

  def to_array(self) -> np.ndarray:
    """Converts the `ConvexPolys` into a rank-1 array."""
    return jnp.concatenate([f.reshape((-1)) for f in dataclasses.astuple(self)])

  def to_normalized_array(self, poly_extents: PolygonExtents) -> jnp.ndarray:
    """Converts the `ConvexPolys` into a rank-1 array with values normalized."""

    cx = utils.normalize(self.center_x, poly_extents.center_x).reshape((-1))
    cy = utils.normalize(self.center_y, poly_extents.center_y).reshape((-1))
    ang = utils.normalize(self.angle_offset, poly_extents.angle_offset).reshape((-1))
    offset = utils.normalize(self.face_offset, poly_extents.face_offset).reshape((-1))

    return jnp.concatenate(( cx, cy, ang, offset ))

  @classmethod
  def from_normalized_array(cls, state_array: jnp.ndarray,
      poly_extents: PolygonExtents)->'ConvexPolys':
    """Converts a normalized rank-1 array into `ConvexPolys`."""
    nb = poly_extents.num_polys
    np = poly_extents.num_planes_in_a_poly

    cx = utils.unnormalize(state_array[0:nb],
                           poly_extents.center_x).reshape((-1))
    cy = utils.unnormalize(state_array[nb:2*nb],
                           poly_extents.center_y).reshape((-1))
    ang = utils.unnormalize(state_array[2*nb:3*nb],
                            poly_extents.angle_offset).reshape((-1))
    offset = utils.unnormalize(state_array[3*nb:],
                               poly_extents.face_offset).reshape((nb, np))
    return ConvexPolys(cx, cy, ang, offset)


def init_poly_grid(nx: int, ny: int, poly_extents: PolygonExtents):
  """
  NOTE: User ensures that the number of polys in `poly_extents` is set as
  `nx*ny`.

  Args:
   nx: number of polys along the x-axis.
   ny: number of polys along the y-axis.
   poly_extents: dataclass of `PolygonExtents` that contain metadata about the
    polys.
  
  Returns: A set of `nx*ny` equi-spaced and equi-sized `ConvexPolys`.
  """
  len_x = np.abs(poly_extents.center_x.max - poly_extents.center_x.min)
  len_y = np.abs(poly_extents.center_y.max - poly_extents.center_y.min)
  del_x = len_x/(4*nx)
  del_y = len_y/(4*ny)
  face_offset = min(del_x, del_y)*np.ones(
      (poly_extents.num_polys, poly_extents.num_planes_in_a_poly))
  cx = poly_extents.center_x.min + np.linspace(2*del_x, len_x - 2*del_x, nx)
  cy = poly_extents.center_y.min + np.linspace(2*del_y, len_y - 2*del_y, ny)
  [cx_grid,cy_grid] = np.meshgrid(cx, cy)
  mean_ang = 0.5*(poly_extents.angle_offset.max + poly_extents.angle_offset.min)
  ang_offset = mean_ang*np.ones((poly_extents.num_polys))
  return ConvexPolys(cx_grid.reshape((-1)), cy_grid.reshape((-1)), 
                    ang_offset, 6*face_offset)


def init_random_polys(poly_extents: PolygonExtents, seed: int = 27):
  """Initialize the polys randomly.

  Args:
    poly_extents: dataclass of `PolygonExtents` that contain metadata about the
    polys.
    seed: Random seed to be used to ensure reproducibility.
  Returns: A set of randomly initialized `ConvexPolys`.
  """
  key = jax.random.PRNGKey(seed)
  cxkey, cykey, angkey, offkey = jax.random.split(key, 4)
  cx = jax.random.uniform(cxkey, (poly_extents.num_polys,),
            minval=poly_extents.center_x.min, maxval=poly_extents.center_x.max)

  cy = jax.random.uniform(cykey, (poly_extents.num_polys,),
            minval=poly_extents.center_y.min, maxval=poly_extents.center_y.max)

  ang = jax.random.uniform(angkey, (poly_extents.num_polys,),
            minval=poly_extents.angle_offset.min,
            maxval=poly_extents.angle_offset.max)

  off = jax.random.uniform(offkey,
                    (poly_extents.num_polys, poly_extents.num_planes_in_a_poly),
            minval=poly_extents.face_offset.min,
            maxval=poly_extents.face_offset.max)
  return ConvexPolys(cx, cy, ang, off)


def compute_poly_sdf(polys: ConvexPolys, mesh: _mesh, order = 100.):
  """
  Compute the signed distance field of the polys onto a mesh. The sdf is the
  Euclidean distance between the boundary of the poly and the mesh elements.
  A negative value indicates that the point is inside the poly and a
  positive value indicates the mesh point lies outside the poly.
  Args:
    polys: A dataclass of `ConvexPolys` that describes a set of polys.
    mesh: describes the mesh onto which the sdf is to be computed.
    order: The entries of logsumexp are roughly [-order, order]. This is
      done to ensure that there is no numerical under/overflow.
  Returns: Array of size (num_polys, num_elems) that is the sdf of each poly
    onto the elements of the mesh.
  """

  # b -> poly, s-> side, e -> element
  relative_x = (mesh.elem_centers[:, 0] - polys.center_x[:, np.newaxis])
  relative_y = (mesh.elem_centers[:, 1] - polys.center_y[:, np.newaxis])
  nrml_dot_x = (jnp.einsum('bs, be-> bse', jnp.cos(polys.face_angles), relative_x) + 
        jnp.einsum('bs, be-> bse', jnp.sin(polys.face_angles), relative_y))
  dist_planes = (nrml_dot_x - polys.face_offset[:, :, np.newaxis])

  # implementation issue: The logsumexp has numerical under/over flow issue. To
  # counter this we scale our distances to be roughly by `order` to be 
  # [-order, order]. We multiply the scaling factor outside of LSE and thus
  # get back the correct SDF. This is purely an implementation trick.
  scaling = mesh.lx/order   # we assume lx and ly are roughly in same order
  sdf = scaling*jax.scipy.special.logsumexp(dist_planes/scaling, axis=1)
  return sdf


def project_sdf_to_density(sdf: jnp.ndarray,
                           mesh: mesher.GridMesher,
                           sharpness: float=10.,
                           order: float=10.
                           ) -> jnp.ndarray:
  """Projects primitive onto a mesh, given primitive parameters and mesh coords.

  The resulting density field has a value of one when an element intersects
  with a primitive and zero when it lies outside the mesh.

  Args:
    sdf: Array of size (num_objects, num_elems) that is the signed distance
      value for each object as for each element on a mesh. 
    sharpness: The sharpness value controls the slope of the sigmoid function.
      While a larger value makes the transition more sharper, it makes it more
      non-linear.
    order: The sigmoid entries are scaled to roughly [-order, order]. This
      is done to prevent the gradients from dying for large magnitudes of
      the entries.
  Returns:
    density: Array of size (num_objects, num_elems) where the values are in
      range [0, 1] where 0 means the mesh element did not intersect with the
      primitive and 1 means it intersected.
  """
  # the sigmoid function has dying gradients for large values of argument.
  # to avoid this we scale it to the order of `order`. Note that simply scaling
  # doesn't shift the 0 isosurface and hence doesn't mess up or calculations.

  scale = mesh.lx/order  # we assume lx and ly are roughly in same order
  scaled_sdf = sdf/scale
  return jax.nn.sigmoid(-sharpness*scaled_sdf)
