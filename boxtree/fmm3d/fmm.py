from boxtree.fmm3d.treeinfo import fmm3d_tree_build
import numpy as np

def reorder(arr, iarr):
    return arr[:, iarr - 1]

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

charge = rng.normal(queue, nparticles, dtype=np.float64).get(
        queue).reshape((1, nparticles))
dipvec = np.asfortranarray(
        rng.normal(queue, (3, nparticles), dtype=np.float64).get(queue))

from boxtree import TreeBuilder
tb = TreeBuilder(ctx)
tree, _ = tb(queue, particles, max_particles_in_box=5)

from boxtree.traversal import FMMTraversalBuilder
tg = FMMTraversalBuilder(ctx)
trav, _ = tg(queue, tree)

ifdipole = 1
ifcharge = 1
ifpgh = 3
ifpghtarg = 3
zk = 0

itree, ipointer, treecenters, boxsize, \
    source, nsource, targ, ntarg, expc, nexpc, \
    isrc, itarg, iexpc, isrcse, itargse, iexpcse \
            = fmm3d_tree_build(tree, trav, queue)

if ifcharge == 0:
    charge = np.array([])

b0 = boxsize[0]
b0inv = 1.0/b0
b0inv2 = b0inv**2
