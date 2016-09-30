"""Routines to create discourse tree objects out of token and parent lists."""

from __future__ import absolute_import
from __future__ import print_function

import itertools
import math

import Queue as Q

import numpy as np


def build_discourse_tree(edu_ids, parent_ids):
  """Build DiscourseTreeNode object from paired (node,parent) IDs for spans."""

  ids_and_parents = zip(edu_ids, parent_ids)
  ids_and_parents.sort(key=lambda t: t[0])
  by_id = itertools.groupby(ids_and_parents, key=lambda t: t[0])
  nodes = []
  ids_to_nodes = {}
  for k, g in by_id:
    g = list(g)
    new_node = DiscourseTreeNode(k, g[0][1], len(g))
    nodes.append(new_node)
    ids_to_nodes[k] = new_node

  for node in nodes:
    if node.parent_id != -1:
      parent_node = ids_to_nodes[node.parent_id]
      node.parent = parent_node
      parent_node.children.append(node)

  root_node = (node for node in nodes if node.parent_id == -1).next()
  for node in discourse_tree_depth_first_walk(root_node):
    if node.parent_id != -1:
      node.parent.total_num_leaves += node.total_num_leaves

  for node in nodes:
    if node.children:
      node.child_sum_tree = build_sum_tree(
          [(n.total_num_leaves + 1, n.node_id) for n in node.children])

  add_tree_levels(root_node)

  return root_node


class DiscourseTreeNode(object):
  """Class representing a discourse-parsed sentence/document."""

  def __init__(self, node_id, parent_id, node_size, span_tokens=None):
    self.node_id = node_id
    self.parent_id = parent_id
    self.node_size = node_size
    self.span_tokens = span_tokens
    self.parent = None
    self.level = None
    self.children = []
    self.child_sum_tree = None
    self.total_num_leaves = node_size

  def tree_num_nodes(self):
    num_nodes = 0
    for c in self.children:
      num_nodes += c.tree_num_nodes()
    return num_nodes + 1


def build_sum_tree(num_leaves_node_id_pairs):
  """Builds tree for fft-tree inference across multiple sibling sets.

  Lay out cousin sibling sets aligned to powers of 2 so that binary-tree
  auxiliary variable model can be masked to keep cousin sums separate.

  Args:
    num_leaves_node_id_pairs: list of (size, id) pairs for nodes.

  Returns:
    SumTreeBranch object describing layout.
  """

  q = Q.PriorityQueue()
  for num_leaves, node_id in num_leaves_node_id_pairs:
    q.put((num_leaves, SumTreeLeaf(node_id, num_leaves)))

  while not q.empty():
    node_a = q.get()
    if q.empty():
      ret = node_a[1]
    else:
      node_b = q.get()
      new_branch = SumTreeBranch(node_a[1], node_b[1])
      q.put((new_branch.width, new_branch))

  return ret


def sum_tree_depth_first_leaf_walk(node):
  if isinstance(node, SumTreeLeaf):
    yield node
  else:
    for n in sum_tree_depth_first_leaf_walk(node.left):
      yield n
    for n in sum_tree_depth_first_leaf_walk(node.right):
      yield n


def round_up_to_power_2(x):
  return 2.0**math.ceil(math.log(x, 2.0))


class SumTreeLeaf(object):

  def __init__(self, span_id, width):
    self.span_id = span_id
    self.update_width(width)

  def update_width(self, new_width):
    self.width = round_up_to_power_2(new_width)
    self.depth = math.log(self.width, 2.0)


class SumTreeBranch(object):

  def __init__(self, left, right):
    self.left = left
    self.right = right
    self.update_width(
        round_up_to_power_2(self.left.width + self.right.width))

  def update_width(self, new_width):
    self.width = new_width
    self.depth = math.log(self.width, 2.0)
    self.left.update_width(new_width / 2)
    self.right.update_width(new_width / 2)


def discourse_tree_depth_first_walk(node):
  for n in node.children:
    for desc in discourse_tree_depth_first_walk(n):
      yield desc
  yield node


def add_tree_levels(tree):
  if tree.parent is None:
    tree.level = 0
  else:
    tree.level = tree.parent.level + 1
  for c in tree.children:
    add_tree_levels(c)


def get_junction_tree_dimensions(examples, tree_cutoff_depth=20):
  """Find dimensions of minimum-sized containing PGM for a set of examples.

  Args:
    examples: list of SummaryExamples.
    tree_cutoff_depth: max depth for BP tree.

  Returns:
    Dimensions of junction tree.
  """

  # take a generator of examples and stream over it to find the
  # size proportions of the junction tree

  max_num_spans = -1
  max_num_sentences = -1
  max_tree_nodes_any_level = -1
  max_tree_widths_at_level = np.zeros([tree_cutoff_depth])
  max_fft_tree_widths_at_level = np.zeros([tree_cutoff_depth])
  global_fft_tree_width = -1

  scratch_tree_widths_at_level = np.zeros([tree_cutoff_depth])
  scratch_fft_tree_widths_at_level = np.zeros([tree_cutoff_depth])
  scratch_nodes_at_level = np.zeros([tree_cutoff_depth])

  for ex in examples:

    trees = [build_discourse_tree(edu_ids, parent_ids)
             for edu_ids, parent_ids in zip(ex.article_edu_ids,
                                            ex.article_parent_ids)]

    max_num_sentences = max(max_num_sentences, len(trees))
    max_num_spans = max(max_num_spans, sum([t.tree_num_nodes() for t in trees]))

    # get per-sentence dimensions
    for tree in trees:
      scratch_tree_widths_at_level[:] = 0
      scratch_fft_tree_widths_at_level[:] = 0
      scratch_nodes_at_level[:] = 0

      all_nodes = list(discourse_tree_depth_first_walk(tree))
      # to layout cousins we have to sort them biggest first
      # but to find the total width we don't need to sort
      for n in all_nodes:
        scratch_tree_widths_at_level[n.level] += n.total_num_leaves + 1
        scratch_nodes_at_level[n.level] += 1
        if n.child_sum_tree is not None:
          scratch_fft_tree_widths_at_level[n.level] += n.child_sum_tree.width

      max_tree_nodes_any_level = max(
          np.max(scratch_nodes_at_level), max_tree_nodes_any_level)
      max_tree_widths_at_level = np.maximum(max_tree_widths_at_level,
                                            scratch_tree_widths_at_level)
      max_fft_tree_widths_at_level = np.maximum(
          max_fft_tree_widths_at_level, scratch_fft_tree_widths_at_level)

    # get global sum tree dimension

    global_sum_tree = build_sum_tree(
        [(dt.total_num_leaves + 1, dt.node_id) for dt in trees])

    global_fft_tree_width = max(global_fft_tree_width, global_sum_tree.width)

  return (max_num_spans, max_num_sentences, max_tree_nodes_any_level,
          max_tree_widths_at_level, max_fft_tree_widths_at_level,
          global_fft_tree_width)
