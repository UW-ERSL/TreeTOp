"""Express constrained optimization as unconstrained loss."""

import dataclasses
from enum import Enum, auto
from typing import Union
import jax.numpy as jnp
import numpy as np

class LossTypes(Enum):
  PENALTY = auto()
  LOG_BARRIER = auto()


@dataclasses.dataclass
class PenaltyParams:
  alpha0: float
  del_alpha: float

  def update_alpha(self, epoch: int):
    self.alpha = self.alpha0 + epoch*self.del_alpha

  def __post_init__(self):
    self.alpha = self.alpha0
    self.loss_type = LossTypes.PENALTY


@dataclasses.dataclass
class LogBarrierParams:
  t0: float
  mu: float

  def update_t(self, epoch: int):
    self.t = self.t0*self.mu**epoch

  def __post_init__(self):
    self.t = self.t0
    self.loss_type = LossTypes.LOG_BARRIER


LossParams = Union[PenaltyParams, LogBarrierParams]

def combined_loss(objective: float,
                  constraints: list[float],
                  loss_params: LossParams,
                  epoch: int):
  """Compute unconstrained loss term for a constrained optimization problem.

      The constraint optimization problem is of type
                min_(x) f(x)
                s.t. g_i(x) <= 0 , i = 1,2,...,N

      NOTE: penalty models the constraints as equality constraints while
        log barrier models as an inequality constraint.

  Args:
    objective: float that is the objective of the problem
    constraints: list of size N of floats that are the constraint values
    loss_type: Supports augmented lagragian, penalty and log barrier methods
    loss_params: parameters that is correspondent of the loss_type
    epoch: integer of the current iteration number.

  Returns: a float that is the combined loss of the objective and constraints
  """
  loss = objective
  if(loss_params.loss_type == LossTypes.PENALTY):
    loss_params.update_alpha(epoch)
    for c in constraints:
      loss = loss + loss_params.alpha*c**2

  if(loss_params.loss_type == LossTypes.LOG_BARRIER):
    loss_params.update_t(epoch)
    t = loss_params.t
    for c in constraints:
      if(c < (-1/t**2)):
        loss = loss - jnp.log(-c)/t
      else:
        loss = loss + (t*c - np.log(1/t**2)/t + 1./t)

  return loss