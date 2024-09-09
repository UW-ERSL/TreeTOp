"""Binary CSG Tree."""

import dataclasses
import enum
from typing import Union, List
import jax.numpy as jnp
import jax
import copy

import utils

_Ext = utils.Extent



class BooleanOperations(enum.Enum):
  INTERSECTION = 0
  UNION = 1
  SUBTRACTION = 2
  SUBTRACTION_REVERSE = 3


def get_operators_from_design_var(design_var: jnp.ndarray,
                                  omega: float=20.,
                                  t: float=1.e1,
                                  ) -> jnp.ndarray:
  """
  Compute the operators from the design variable.

  We follow along the paper "A Unified Differentiable Boolean Operator with 
    Fuzzy Logic, Liu, 2024, SIGGRAPH" to derive the operators from the
    design variables.
  
  The design variables are arrays with values in [0,1] that are random. They are
    to be transformed into one-hot encodings for each operator. This is done as:

                c = softmax(sin(w*design_var) * t)

    where w (`omega`) is a frequency term that makes changing operators smooth.
    Further, `t` is a temperature term that makes the operators more closer to 
    a one-hot encoding.

  Args:
    design_var: A 1D array of shape (num_operators, 4) containing the design
      variables that are to be transformed to the operators.
    omega: The scaling factor of the design variable.
    t: The temperature of the design variable.
  Returns: Array of (num_operators, 4) containing the one-hot encodings of the
    operators.
  """
  return jax.nn.softmax(t*jnp.sin(omega*design_var), axis=1)


def unified_boolean_operator(dens_x: jnp.ndarray,
                             dens_y: jnp.ndarray,
                             c: jnp.ndarray
                             ) -> jnp.ndarray:
  """
  Compute the unified boolean operator.
    The unified boolean operator is defined as:
          B_c(x, y) = (c_1 + c_2)x + (c_1 + c_3)y + (c_0 - c_1 - c_2 - c_3)xy
    where c_i are the coefficients of the unified boolean operator.
            c : (1, 0, 0, 0) -> intersection
            c : (0, 1, 0, 0) -> union
            c : (0, 0, 1, 0) -> x - y
            c : (0, 0, 0, 1) -> y - x
  Args:
    dens_x: A 1D array of shape (num_pts) containing the first fuzzy 
        occupancy function with values in [0,1].
    dens_x: A 1D array of shape (num_pts) containing the second fuzzy
        occupancy function with values in [0,1].
    c: A 1D array of shape (4,) containing the coefficients of the
       unified boolean operator.
  Returns:
    A 1D array of shape (num_pts,) containing the unified boolean shapes.
  """
  return (
          (c[1] + c[2]) * dens_x +
          (c[1] + c[3]) * dens_y +
          (c[0] - c[1] - c[2] - c[3]) * dens_x * dens_y
          )


@dataclasses.dataclass
class Node:
  left: Union[None, 'Node'] = None
  right: Union[None, 'Node'] = None
  operation: Union[None, jnp.ndarray] = None
  value: Union[None, jnp.ndarray] = None
  is_redundant: bool = False


def init_csg_tree(primitive_density: jnp.ndarray,
                  operations: jnp.ndarray)->Node:
  """
  Build a binary tree from the leaf values assuming it's a perfect binary tree.

  Args:
    primitive_density: Array of (2^n, num_pts) of the occupancy functions of the 
      primitives at the 0th level at `num_pts` points. There are a total of 
      2^n primitives at the 0th level that are subsequently combined using 
      boolean operations.
    operations: Array of (2^n - 1, 4) of the boolean operations to be applied
      to the primitives. The operations are in post-traversal order.
  """
  if not primitive_density.size:
    return None

  # Create leaf nodes
  leaves = [Node(value=value) for value in primitive_density]

  # Build the tree bottom-up
  while len(leaves) > 1:
    parents = []
    for i in range(0, len(leaves), 2):
      parent = Node()
      parent.left = leaves[i]
      parent.right = leaves[i + 1]
      parents.append(parent)
    leaves = parents

  populate_operations(leaves[0], operations, 0)
  return leaves[0]


def populate_operations(node: Node,
                        operation: jnp.ndarray,
                        index: int) -> int:
  """
  Traverse the tree in post-order and populate non-leaf nodes.
  The function uses indexing to populate values.

  Args:
    node: The root node of the tree.
    operations: Array of (2^n - 1, 4) of the boolean operations to be applied
      to the primitives. The operations are in post-traversal order.
    index: The current level of the tree. Defaults to 0 (root level).
  """
  if not node:
    return index

  index = populate_operations(node.left, operation, index)
  index = populate_operations(node.right, operation, index)

  if node.left is not None and node.right is not None:
    # Populate the node's value from the numpy array using the current index
    node.operation = operation[index, :]
    index += 1

  return index


def perform_csg_operations(node: Union[Node, None]):
  """
  Traverse the tree in post-order and populate non-leaf nodes.
  Args:
    node: Initalized with the root node of the tree. The tree is traversed in
      post-order and intermediate nodes are passed when called recursively.
  """
  if not node:
    return

  perform_csg_operations(node.left)
  perform_csg_operations(node.right)
  
  if node.left is not None and node.right is not None:
    node.value = unified_boolean_operator(node.left.value,
                                          node.right.value,
                                          node.operation)
  return


def eval_binary_csg_tree(primitive_density: jnp.ndarray,
                     operations: jnp.ndarray
                     )->Node:
  """Evaluates a binary Constructive Solid Geometry (CSG) tree using the provided primitive densities and operations.
    Args:
      primitive_density : The array representing the density values of the primitive shapes.
      operations : The array of operations to be performed on the CSG tree nodes. 
    Returns:
      Node: The root node/final design of the evaluated CSG tree after performing all operations.
    """
  root = init_csg_tree(primitive_density, operations)
  perform_csg_operations(root)
  return root


def breadth_first_search_at_depth(root: Node, depth: int)->List[Node]:
  """Perform a breadth-first search at a given depth of the tree.
  Args:
    root: The root node of the tree.
    depth: The depth at which the search is to be performed.
  Returns:
    A list of nodes at the given depth. If the depth is invalid, an empty list is
    returned.
  """
  if not root or depth < 0:
    return []

  result = []
  queue = [(root, 0)]  # Store nodes with their depths

  while queue:
    node, current_depth = queue.pop(0)

    if current_depth == depth:
      result.append(node)
    elif current_depth < depth:
      if node.left:
        queue.append((node.left, current_depth + 1))
      if node.right:
        queue.append((node.right, current_depth + 1))

  return result


def get_all_child_nodes(node: Union[Node, None])->List[Node]:
  """
  Recursively collects all child nodes of a given node in a perfectly 
    balanced binary tree.

  Args:
    node: The node from which to start collecting child nodes.

  Returns:
    A list containing all child nodes (including descendants) of the given node.
    An empty list is returned if the node contains no children/descendants.
  """

  if node is None:
    return []

  child_nodes = []

  child_nodes.extend(get_all_child_nodes(node.left))
  child_nodes.extend(get_all_child_nodes(node.right))

  if node.left:
    child_nodes.append(node.left)
  if node.right:
    child_nodes.append(node.right)

  return child_nodes


def prune_tree(node: Union[Node, None],
              num_max_px_diff: int= 5)->None:
  """Mark the nodes that are redundant and can be pruned from the tree.
  Args:
    node: The root node of the tree.
    num_max_px_diff: The maximum number of pixel differences allowed between the
      actual node value and the value obtained by replacing either of its children
      with ones or zeros. If the difference is less than or equal to this value,
      the node is marked as redundant.
  Returns: None
  """
  if not node:
    return

  prune_tree(node.left)
  prune_tree(node.right)
  
  def _mark_children_redundant(children: List[Node])->None:
    for child in children:
      child.is_redundant = True

  if node.left is not None and node.right is not None:
    left_val = node.left.value
    right_val = node.right.value
    node.value = jnp.round(unified_boolean_operator(node.left.value,
                                          node.right.value,
                                          node.operation))
    node_val_left_1 = jnp.round(unified_boolean_operator(jnp.ones_like(left_val),
                                               node.right.value,
                                                node.operation))
    node_val_right_1 = jnp.round(unified_boolean_operator(node.left.value,
                                                jnp.ones_like(right_val),
                                                node.operation))
    node_val_left_0 = jnp.round(unified_boolean_operator(jnp.zeros_like(left_val),
                                                node.right.value,
                                                  node.operation))
    node_val_right_0 = jnp.round(unified_boolean_operator(node.left.value,
                                                jnp.zeros_like(right_val),
                                                node.operation))
    
    delta_left_1 = jnp.sum(jnp.abs(node.value - node_val_left_1))
    delta_right_1 = jnp.sum(jnp.abs(node.value - node_val_right_1))
    delta_left_0 = jnp.sum(jnp.abs(node.value - node_val_left_0))
    delta_right_0 = jnp.sum(jnp.abs(node.value - node_val_right_0))

    if ( delta_left_1 <= num_max_px_diff or delta_left_0 <= num_max_px_diff):
      _mark_children_redundant(get_all_child_nodes(node.left))
      node.left.is_redundant = True

    if ( delta_right_1 <= num_max_px_diff or delta_right_0 <= num_max_px_diff):
      _mark_children_redundant(get_all_child_nodes(node.right))
      node.right.is_redundant = True

  return


