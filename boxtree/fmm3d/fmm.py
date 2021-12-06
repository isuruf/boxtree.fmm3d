from boxtree.fmm3d.treeinfo import fmm3d_tree_build
from boxtree.fmm3d.fortran import l3dterms, h3dterms, mpalloc
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
eps = 1e-10
# number of fmms
nd = 1

itree, ipointer, treecenters, boxsize, \
    source, nsource, targ, ntarg, expc, nexpc, \
    isrc, itarg, iexpc, isrcse, itargse, iexpcse \
            = fmm3d_tree_build(tree, trav, queue)

nlevels = tree.nlevels - 1
nboxes = tree.nboxes

if ifcharge == 0:
    charge = np.array([])

b0 = boxsize[0]
b0inv = 1.0/b0
b0inv2 = b0inv**2
b0inv3 = b0inv2*b0inv

# src/Laplace/lfmm3d.f#L412
sourcesort = reorder(source, isrc)
sourcesort *= b0inv

# src/Laplace/lfmm3d.f#L419
if ifcharge == 1:
    chargesort = reorder(charge, isrc)
    chargesort *= b0inv
else:
    chargesort = np.array([])

# src/Laplace/lfmm3d.f#L426
if ifdipole == 1:
    dipvecsort = reorder(dipvec, isrc)
    dipvecsort *= b0inv2
else:
    dipvecsort = np.array([])

# src/Laplace/lfmm3d.f#L435
targsort = reorder(source, itarg)
targsort *= b0inv

# src/Laplace/lfmm3d.f#L442
treecenters *= b0inv
boxsize *= b0inv

# src/Helmholtz/hfmm3d.f#L411
zkfmm = np.complex128(zk * b0)

laplace = (zk == 0)

if laplace:
    # src/Laplace/lfmm3d.f#L467
    scales = boxsize
else:
    # src/Helmholtz/hfmm3d.f#L418
    scales = boxsize * abs(zkfmm)
    scales[scales > 1] = 1

nterms = np.empty(nlevels + 1, dtype=np.int32)
for i in range(nlevels + 1):
    if laplace:
        # src/Laplace/lfmm3d.f#L392
        l3dterms(eps, nterms[i])
    else:
        # src/Helmholtz/hfmm3d.f#L428
        h3dterms(boxsize[i], zkfmm, eps, nterms[i])

nmax = np.max(nterms)
lmptemp = (nmax+1)*(2*nmax+1)*2*nd
iaddr = np.empty((2, nboxes), dtype=np.int64, order='F')
mptemp = np.empty(lmptemp, dtype=np.double)
mptemp2 = np.empty(lmptemp, dtype=np.double)
lmptot = np.zeros(1, dtype=np.int64)

laddr = itree[0: 2*(nlevels+1)].reshape(2, (nlevels + 1), order='F')
mpalloc(nd=np.int32(nd), laddr=laddr, iaddr=iaddr,
        nlevels=np.int32(nlevels), lmptot=lmptot, nterms=nterms)

assert lmptot[0] != 0
rmlexp = np.empty(lmptot, dtype=np.double)
