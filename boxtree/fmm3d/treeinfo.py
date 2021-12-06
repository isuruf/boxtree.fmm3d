import numpy as np
from boxtree.fmm3d.fortran import pts_tree_sort

def fmm3d_tree_build(tree, trav, queue):
    nlevels = tree.nlevels
    nboxes = tree.nboxes

    box_levels = tree.box_levels.get(queue)
    box_parent_ids = tree.box_parent_ids.get(queue)
    box_child_ids = tree.box_child_ids.get(queue)[:, :nboxes]
    coll_starts = trav.colleagues_starts.get(queue)
    coll_lists = trav.colleagues_lists.get(queue)
    box_centers = tree.box_centers.get(queue)[:, :nboxes]

    iptr = np.zeros(8, dtype=np.int64)
    iptr[0] = 1
    iptr[1] = 2*(nlevels+1) + 1
    iptr[2] = iptr[1] + nboxes
    iptr[3] = iptr[2] + nboxes
    iptr[4] = iptr[3] + nboxes
    iptr[5] = iptr[4] + 8*nboxes
    iptr[6] = iptr[5] + nboxes
    iptr[7] = iptr[6] + 27*nboxes

    ltree = iptr[7] - 1
    assert ltree == 39*nboxes + 2*(nlevels+1)

    itree = np.zeros(ltree, dtype=np.int32)

    for i in range(nlevels):
        # first box for ith level
        itree[2*i] = tree.level_start_box_nrs[i] + 1
        # last box for ith level
        itree[2*i + 1] = tree.level_start_box_nrs[i + 1]

    # level of the boxes
    itree[iptr[1] - 1:iptr[2] - 1] = box_levels

    # parent of the boxes
    itree[iptr[2] - 1:iptr[3] - 1] = box_parent_ids
    itree[iptr[2] - 1] = -1   # for box 0, set -1

    # number of childs of boxes
    itree[iptr[3] - 1:iptr[4] - 1] = np.count_nonzero(box_child_ids, axis=0)

    # child ids of boxes
    itree[iptr[4] - 1:iptr[5] - 1] = -1
    for i in range(nboxes):
        child_boxes = box_child_ids[:, i]
        non_zero_child_boxes = child_boxes[np.nonzero(child_boxes)]
        istart = iptr[4] + 8*i - 1
        itree[istart:istart + len(non_zero_child_boxes)] = non_zero_child_boxes + 1

    # ncolleagues
    itree[iptr[5] - 1:iptr[6] - 1] = coll_starts[1:] - coll_starts[:-1]

    # icolleagues
    itree[iptr[6] - 1:iptr[7] - 1] = -1
    for i in range(nboxes):
        istart = iptr[6] + 27*i - 1
        itree[istart:istart+itree[iptr[5] + i - 1]] = \
                coll_lists[coll_starts[i]:coll_starts[i+1]] + 1

    boxsize = np.zeros(nlevels + 1, dtype=np.int32)
    boxsize[0] = tree.root_extent
    for i in range(nlevels):
        boxsize[i + 1] = boxsize[i] / 2

    nexpc = 0
    expc = np.zeros((3, nexpc), dtype=np.double, order='F')

    source = np.array([row.get(queue) for row in tree.sources], order='F')
    nsource = source.shape[1]

    targ = np.array([row.get(queue) for row in tree.targets], order='F')
    ntarg = targ.shape[1]

    isrc = np.zeros(nsource, dtype=np.int32)
    itarg = np.zeros(ntarg, dtype=np.int32)
    iexpc = np.zeros(1, dtype=np.int32)

    isrcse = np.zeros((2, nboxes), dtype=np.int32, order='F')
    itargse = np.zeros((2, nboxes), dtype=np.int32, order='F')
    iexpcse = np.zeros((2, nboxes), dtype=np.int32, order='F')

    treecenters = np.asfortranarray(box_centers)

    pts_tree_sort_kwargs = dict(
        itree=itree,
        ltree=ltree,
        nboxes=nboxes,
        nlevels=nlevels,
        iptr=iptr,
        centers=treecenters,
    )

    pts_tree_sort(n=nsource, xys=source, ixy=isrc, ixyse=isrcse,
        **pts_tree_sort_kwargs)

    pts_tree_sort(n=ntarg, xys=targ, ixy=itarg, ixyse=itargse,
        **pts_tree_sort_kwargs)

    pts_tree_sort(n=nexpc, xys=expc, ixy=iexpc, ixyse=iexpcse,
        **pts_tree_sort_kwargs)

    return itree, iptr, treecenters, boxsize, \
        source, nsource, targ, ntarg, expc, nexpc, \
        isrc, itarg, iexpc, isrcse, itargse, iexpcse


import pyopencl as cl
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

dims = 3
nparticles = 500

from pyopencl.clrandom import RanluxGenerator
rng = RanluxGenerator(queue, seed=15)

from pytools.obj_array import make_obj_array
particles = make_obj_array([
    rng.normal(queue, nparticles, dtype=np.float64)
    for i in range(dims)])

from boxtree import TreeBuilder
tb = TreeBuilder(ctx)
tree, _ = tb(queue, particles, max_particles_in_box=5)

from boxtree.traversal import FMMTraversalBuilder
tg = FMMTraversalBuilder(ctx)
trav, _ = tg(queue, tree)

itree, ipointer, treecenters, boxsize, \
    source, nsource, targ, ntarg, expc, nexpc, \
    isrc, itarg, iexpc, isrcse, itargse, iexpcse \
            = fmm3d_tree_build(tree, trav, queue)

