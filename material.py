
import dataclasses
from typing import Optional
import numpy as np
import jax.numpy as jnp


@dataclasses.dataclass
class Material:
  """ Linear elasticity material constants.
  Attributes:
    youngs_modulus: The young's modulus of the material.
    poissons_ratio: The poisson's ratio of the material.
    delta_youngs_modulus: A small epsilon value of the void material. This is
      added to ensure numerical stability during finite element analysis.
  """
  youngs_modulus: float = 1.
  poissons_ratio: float = 0.3
  delta_youngs_modulus: Optional[float] = 1e-3


def compute_simp_material_modulus(density: jnp.ndarray,
                                  mat_const: Material,
                                  penal: float = 3.,
                                  young_min: float = 1e-3)->jnp.ndarray:
  """
    E = rho_min + E0*( density)^penal
  Args:
    density: Array of size (num_elems,) with values in range [0,1]
    penal: SIMP penalization constant, usually assumes a value of 3
    young_min: Small value added to the modulus to prevent matrix singularity
  Returns: Array of size (num_elems,) which contain the penalized modulus
    at each element
  """
  return young_min + mat_const.youngs_modulus*(density**penal)


def projection_filter(density: jnp.ndarray,
                      beta: float,
                      eta: float=0.5
                      )->jnp.ndarray:
  """Threshold project the density, pushing the values towards 0/1.

  Args:
    density: Array of size (num_elems,) that are in [0,1] that contain the
      density of the elements.
    beta: Sharpness of projection (typically ~ 1-32). Larger value indicates
      a sharper projection.
    eta: Center value about which the values are projected.

  Returns: The thresholded density value array of size (num_elems,).
  """
  v1 = np.tanh(eta*beta)
  nm = v1 + jnp.tanh(beta*(density-eta))
  dnm = v1 + jnp.tanh(beta*(1.-eta))
  return nm/dnm