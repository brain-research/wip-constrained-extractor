"""Uses matplotlib to generate visualizations of attention over a document.

Usage: Given a list of word tokens and a list of floats representing a set of
  attention weights over these tokens (from [0.0, 1.0]), renders a visualization
  of this attention to an image using word highlighting. This image can be
  retrieved as a numpy array using draw_to_numpy_array or written to disk
  using write_image.
"""

import io

import matplotlib

import matplotlib.patches as ps
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np


font_opts = {}

matplotlib.use("TkAgg")


def write_image(path, tokens, weights):
  """Render attention diagram and write to file.

  Args:
    path: path at which to write file.
    tokens: list of string tokens.
    weights: list of float attention weights in [0.0,1.0] for tokens.
  """

  f = render_attn_inner(tokens, weights)
  f.savefig(path, bbox_inches="tight", frameon=False)
  plt.close(f)


def render_attn(tokens, weights):
  """Render attention diagram to a numpy array.

  Args:
    tokens: list of string tokens.
    weights: list of float attention weights in [0.0,1.0] for tokens.

  Returns:
    f: numpy array containing pixels of rendered diagram.
  """

  f = render_attn_inner(tokens, weights)
  arr = draw_to_numpy_array(f)
  plt.close(f)
  return arr


def draw_to_numpy_array(fig):
  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
  data_wxhx3 = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return data_wxhx3


def render_attn_inner(tokens, weights):
  """Render attention diagram to a matplotlib canvas.

  Args:
    tokens: list of string tokens.
    weights: list of float attention weights in [0.0,1.0] for tokens.

  Returns:
    f: matplotlib fig containing rendered diagram.
  """

  plt.figure(figsize=(8, 8), dpi=80)
  plt.margins(0.0)

  sizes, line_height, space_widths = get_sizes(tokens)

  # First pass layout to get line breaks with rough whitespace
  boxes, line_breaks, line_poss = layout_bounding_boxes(
      0.05, 0.95, 0.9, line_height, space_widths, 0.005, sizes)

  f, ax = plt.subplots()
  r = f.canvas.get_renderer()

  line_toks = split_subsequences(tokens, line_breaks)
  line_spaces = split_subsequences(space_widths, line_breaks)
  line_tok_sizes = split_subsequences(sizes, line_breaks)

  # Now rescale spaces on individual lines depending on what tokens are on them
  rescaled_spaces = []
  for line, line_pos, spaces, tok_sizes in zip(line_toks, line_poss,
                                               line_spaces, line_tok_sizes):
    line_text = " ".join(line)
    line_width, line_height = measure_text(line_text, r, ax)

    t = plt.text(line_pos[0], line_pos[1], line_text, **font_opts)

    line_width, line_height = measure_text_obj(t, r, ax)

    rect = ps.Rectangle(
        [line_pos[0], line_pos[1]],
        line_width,
        line_height,
        color=[0, 0, 1],
        fill=True,
        alpha=0.2)

    old_line_width = sum([w for w, _ in tok_sizes]) + sum(spaces[:-1])

    ratio = (line_width) / old_line_width
    rescaled_spaces.extend([s * ratio for s in spaces])

  # Final layout pass with the rescaled spaces
  boxes, line_breaks, line_poss = layout_bounding_boxes(
      0.05, 0.95, 0.9, line_height, rescaled_spaces, 0.005, sizes)

  for _, box, w in zip(tokens, boxes, weights):
    pad = 0.004
    rect = ps.Rectangle(
        [box[0] - pad, box[1]],
        box[2] + 2 * pad,
        box[3],
        color=[0, 0, 1],
        fill=True,
        alpha=0.5 * w)
    ax.add_patch(rect)

  ax.patch.set_facecolor("white")
  f.patch.set_facecolor("white")

  ax.axis("off")

  return f


def find_renderer(fig):
  """Hacks to find renderers for matplotlib in different settings."""
  if hasattr(fig.canvas, "get_renderer"):
    renderer = fig.canvas.get_renderer()
  else:
    fig.canvas.print_pdf(io.BytesIO())
    renderer = fig._cachedRenderer  # pylint: disable=protected-access
  return renderer


def measure_text_obj(t, r, ax):
  """Measure size of matplotlib text object on canvas."""
  bb = t.get_window_extent(renderer=r)
  inv = ax.transData.inverted()
  bb = transforms.Bbox(inv.transform(bb))
  return (bb.width, bb.height)


def measure_text(text, r, ax):
  """Measure size of text string on canvas."""
  t = plt.text(0.5, 0.5, text, **font_opts)
  res = measure_text_obj(t, r, ax)
  t.remove()
  return res


def get_sizes(tokens):
  """Measures the size of a list of tokens when drawn on the canvas.

  Args:
    tokens: list of token strings

  Returns:
    bboxes: list of bounding boxes for each token
    line_height: height of the rendered line
    space_widths: widths for each space between words
  """

  f, ax = plt.subplots()
  r = find_renderer(f)
  boxes = []
  for tok in tokens + [" ".join(tokens)]:
    box = measure_text(tok, r, ax)
    boxes.append(box)
  bboxes = boxes[:-1]
  line_height = boxes[-1][1]
  avg_space_width = (boxes[-1][0] - sum([w for w, _ in bboxes])) / (
      len(tokens) - 1)
  space_widths = []
  for fst, fstbox, snd, sndbox in zip(tokens[:-1], bboxes[:-1], tokens[1:],
                                      bboxes[1:]):
    pair = fst + " " + snd
    box = measure_text(pair, r, ax)
    fstbox = measure_text(fst, r, ax)
    sndbox = measure_text(snd, r, ax)
    space_width = box[0] - fstbox[0] - sndbox[0]
    space_widths.append(space_width)
  avg_pairwise_space_width = sum(space_widths) / len(space_widths)
  ratio = avg_space_width / avg_pairwise_space_width
  space_widths = [w * ratio for w in space_widths]
  # add a dummy final space
  space_widths.append(0.0)
  return bboxes, line_height, space_widths


def layout_bounding_boxes(canvas_x, canvas_y, canvas_width, line_height,
                          space_widths, y_space, sizes):
  """Layout token bounding boxes on canvas with greedy wrapping.

  Args:
    canvas_x: x coordinate of top left of canvas
    canvas_y: y coordinate of top left of canvas
    canvas_width: width of canvas
    line_height: height for each line
    space_widths: width for space between each word
    y_space: extra space between lines
    sizes: list of width,height tuples of sizes for each word

  Returns:
    boxes: 4-tuple bounding box for each word
    line_breaks: token index for each line break
    line_poss: x,y positions starting each line
  """

  cur_x, cur_y = canvas_x, canvas_y
  cur_size = 0
  cur_line = 0
  boxes = []
  line_breaks = []
  line_poss = []
  line_poss.append((cur_x, cur_y))
  while cur_size < len(sizes):
    sz = sizes[cur_size]
    if cur_x + sz[0] > canvas_width + canvas_x:
      cur_line += 1
      cur_y = canvas_y - cur_line * (y_space + line_height)
      cur_x = canvas_x
      line_poss.append((cur_x, cur_y))
      line_breaks.append(cur_size)
    else:
      boxes.append((cur_x, cur_y, sz[0], sz[1]))
      cur_x += sz[0]
      if cur_size < len(space_widths):
        cur_x += space_widths[cur_size]
      cur_size += 1
  return boxes, line_breaks, line_poss


def split_subsequences(sequence, sub_lengths):
  """Splits a list into sub-lists with given lengths."""

  line_toks = []
  cur_line_toks = []
  cur_line = 0
  for i in xrange(len(sequence)):
    if cur_line < len(sub_lengths):
      cur_break = sub_lengths[cur_line]
      if i >= cur_break:
        cur_line += 1
        line_toks.append(cur_line_toks)
        cur_line_toks = []
    cur_tok = sequence[i]
    cur_line_toks.append(cur_tok)
  line_toks.append(cur_line_toks)
  return line_toks
