import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

dims = 3
nparticles = 500

# -----------------------------------------------------------------------------
# generate some random particle positions
# -----------------------------------------------------------------------------
from pyopencl.clrandom import RanluxGenerator
rng = RanluxGenerator(queue, seed=15)

from pytools.obj_array import make_obj_array
particles = make_obj_array([
    rng.normal(queue, nparticles, dtype=np.float64)
    for i in range(dims)])

# -----------------------------------------------------------------------------
# build tree and traversals (lists)
# -----------------------------------------------------------------------------
from boxtree import TreeBuilder
tb = TreeBuilder(ctx)
tree, _ = tb(queue, particles, max_particles_in_box=5)

from boxtree.traversal import FMMTraversalBuilder
tg = FMMTraversalBuilder(ctx)
trav, _ = tg(queue, tree)

box_levels = tree.box_levels.get(queue)
box_parent_ids = tree.box_parent_ids.get(queue)
box_child_ids = tree.box_child_ids.get(queue)[:, :nboxes]
coll_starts = trav.colleagues_starts.get(queue)
coll_lists = trav.colleagues_lists.get(queue)

nlevels = tree.nlevels
nboxes = tree.nboxes

iptr = [0]*8
iptr[0] = 0
iptr[1] = 2*(nlevels+1)
iptr[2] = iptr[1] + nboxes
iptr[3] = iptr[2] + nboxes
iptr[4] = iptr[3] + nboxes
iptr[5] = iptr[4] + 8*nboxes
iptr[6] = iptr[5] + nboxes
iptr[7] = iptr[6] + 27*nboxes

ltree = iptr[7]
assert ltree == 39*nboxes + 2*(nlevels+1)

itree = np.zeros(ltree, dtype=np.int)

# first box for first level
itree[0] = 0
# last box for first level
itree[1] = 0

# level of the boxes
itree[iptr[1]:iptr[2]] = box_levels

# parent of the boxes
itree[iptr[2]:iptr[3]] = box_parent_ids
itree[iptr[2]] = -1   # for box 0, set -1

# number of childs of boxes
itree[iptr[3]:iptr[4]] = np.count_nonzero(box_child_ids, axis=0)

# child ids of boxes
itree[iptr[4]:iptr[5]] = -1
for i in range(nboxes):
    child_boxes = box_child_ids[:, i]
    non_zero_child_boxes = child_boxes[np.nonzero(child_boxes)]
    istart = iptr[4] + 8*i
    itree[istart:istart + len(non_zero_child_boxes)] = non_zero_child_boxes

# ncolleagues
itree[iptr[5]:iptr[6]] = coll_starts[1:] - coll_starts[:-1]

# icolleagues
itree[iptr[6]:iptr[7]] = -1
for i in range(nboxes):
    istart = iptr[6] + 27*i
    itree[istart:istart+itree[iptr[5]+i]] = \
            coll_lists[coll_starts[i]:coll_starts[i+1]]

boxsize = [0]*(nlevels + 1)
boxsize[0] = tree.root_extent
for i in range(nlevels):
    boxsize[i + 1] = boxsize[i] / 2
