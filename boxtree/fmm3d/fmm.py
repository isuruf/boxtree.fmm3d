from boxtree.fmm3d.treeinfo import fmm3d_tree_build
from boxtree.fmm3d.fortran import (l3dterms, h3dterms, mpalloc,
                                   lfmm3dmain, hfmm3dmain)
import numpy as np


def reorder(arr, iarr):
    return arr[..., iarr - 1]


def reorder_inv(arr, iarr):
    res = arr.copy()
    res[..., iarr - 1] = arr
    return res


import pyopencl as cl
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

dims = 3
nparticles = 500

from numpy.random import default_rng
np_rng = default_rng(10)
vals = np_rng.random((3, nparticles - 2), dtype=np.double)
vals = np_rng.dirichlet((10, 5, 2), (nparticles - 2)).T
particles_np = [
    np.append(vals[0], [0.999999, 0.000001]),
    np.append(vals[1], [0.999999, 0.000001]),
    np.append(vals[2], [0.999999, 0.000001]),
]

from pytools.obj_array import make_obj_array
particles = make_obj_array([
    cl.array.to_device(queue, particles_np[i])
    for i in range(dims)])

from pyopencl.clrandom import PhiloxGenerator
rng = PhiloxGenerator(ctx, seed=15)
charge = rng.normal(queue, nparticles, dtype=np.float64).get(
        queue).reshape((1, nparticles))
dipvec = np.asfortranarray(
        rng.normal(queue, (1, 3, nparticles), dtype=np.float64).get(queue))

ndiv = 40
from boxtree import TreeBuilder
tb = TreeBuilder(ctx)
tree, _ = tb(queue, particles, max_particles_in_box=ndiv, kind='adaptive',
             bbox=np.array([[0, 1], [0, 1], [0, 1]], dtype=np.double))

from boxtree.traversal import FMMTraversalBuilder
tg = FMMTraversalBuilder(ctx)
trav, _ = tg(queue, tree)

ifdipole = 0
ifcharge = 1
ifpgh = 1
ifpghtarg = 0
zk = 0
eps = 1e-5
# number of fmms
nd = 1
# flag for periodic implmentations. Currently unused
iper = 0
ifnear = 1

itree, ltree, ipointer, treecenters, boxsize, \
    source, nsource, targ, ntarg, expc, nexpc, \
    isrc, itarg, iexpc, isrcse, itargse, iexpcse, \
    nlevels, nboxes = fmm3d_tree_build(tree, trav, queue)

if ifcharge == 0:
    charge = np.array([])

b0 = boxsize[0]
b0inv = 1.0/b0
b0inv2 = b0inv**2
b0inv3 = b0inv2*b0inv

# src/Laplace/lfmm3d.f#L264-L274
if ifpgh == 1:
    potsort = np.zeros((nd, nsource), dtype=np.double, order='F')
    gradsort = np.zeros((nd, 3, 1), dtype=np.double, order='F')
    hesssort = np.zeros((nd, 6, 1), dtype=np.double, order='F')
elif ifpgh == 2:
    potsort = np.zeros((nd, nsource), dtype=np.double, order='F')
    gradsort = np.zeros((nd, 3, nsource), dtype=np.double, order='F')
    hesssort = np.zeros((nd, 6, 1), dtype=np.double, order='F')
elif ifpgh == 3:
    potsort = np.zeros((nd, nsource), dtype=np.double, order='F')
    gradsort = np.zeros((nd, 3, nsource), dtype=np.double, order='F')
    hesssort = np.zeros((nd, 6, nsource), dtype=np.double, order='F')
else:
    raise ValueError(f"unknown ifpgh value: {ifpgh}")

# src/Laplace/lfmm3d.f#L276-288
if ifpghtarg == 1:
    pottargsort = np.zeros((nd, ntarg), dtype=np.double, order='F')
    gradtargsort = np.zeros((nd, 3, 1), dtype=np.double, order='F')
    hesstargsort = np.zeros((nd, 6, 1), dtype=np.double, order='F')
elif ifpghtarg == 2:
    pottargsort = np.zeros((nd, ntarg), dtype=np.double, order='F')
    gradtargsort = np.zeros((nd, 3, ntarg), dtype=np.double, order='F')
    hesstargsort = np.zeros((nd, 6, 1), dtype=np.double, order='F')
elif ifpghtarg == 3:
    pottargsort = np.zeros((nd, ntarg), dtype=np.double, order='F')
    gradtargsort = np.zeros((nd, 3, ntarg), dtype=np.double, order='F')
    hesstargsort = np.zeros((nd, 6, ntarg), dtype=np.double, order='F')
elif ifpghtarg == 0:
    pottargsort = np.zeros((nd, 1), dtype=np.double, order='F')
    gradtargsort = np.zeros((nd, 3, 1), dtype=np.double, order='F')
    hesstargsort = np.zeros((nd, 6, 1), dtype=np.double, order='F')
else:
    raise ValueError(f"unknown ifpghtarg value: {ifpghtarg}")

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
    dipvecsort = np.zeros((nd, 3, 0), dtype=np.double, order='F')

# src/Laplace/lfmm3d.f#L435
targsort = reorder(targ, itarg)
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
        l3dterms(eps, nterms[i:i+1])
    else:
        # src/Helmholtz/hfmm3d.f#L428
        h3dterms(boxsize[i], zkfmm, eps, nterms[i:i+1])

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
# src/Laplace/lfmm3d.f#L454
rmlexp = np.empty(lmptot, dtype=np.double)

# src/Laplace/lfmm3d.f#L159-L160
texpssort = np.empty((nd, 1, 1, 1), dtype=np.complex128)
expcsort = np.empty((3, nexpc), dtype=np.double)
scjsort = np.empty(1, dtype=np.double)
radssort = np.empty(1, dtype=np.double)

ier = np.zeros(1, dtype=np.int32) * -1

fmm3dmain_kwargs = dict(
    nd=np.int32(nd),
    eps=np.double(eps),
    nsource=np.int32(nsource),
    sourcesort=sourcesort,
    ifcharge=np.int32(ifcharge),
    chargesort=chargesort,
    ifdipole=np.int32(ifdipole),
    dipvecsort=dipvecsort,
    ntarg=np.int32(ntarg),
    targsort=targsort,
    nexpc=np.int32(nexpc),
    expcsort=expcsort,
    iaddr=iaddr,
    rmlexp=rmlexp,
    lmptot=lmptot,
    mptemp=mptemp,
    mptemp2=mptemp2,
    lmptemp=lmptemp,
    itree=itree,
    ltree=np.int32(ltree),
    ipointer=ipointer,
    ndiv=np.int32(ndiv),  # not used
    nlevels=np.int32(nlevels),
    nboxes=np.int32(nboxes),
    iper=np.int32(iper),
    boxsize=boxsize,
    centers=treecenters,
    isrcse=isrcse,
    itargse=itargse,
    iexpcse=iexpcse,
    rscales=scales,
    laddr=laddr,
    nterms=nterms,
    ifpgh=np.int32(ifpgh),
    pot=potsort,
    grad=gradsort,
    hess=hesssort,
    ifpghtarg=ifpghtarg,
    pottarg=pottargsort,
    gradtarg=gradtargsort,
    hesstarg=hesstargsort,
    ntj=np.int32(0),
    scjsort=scjsort,
    ifnear=np.int32(1),
    ier=ier)

assert ier == 0

if laplace:
    lfmm3dmain(
        tsort=texpssort,
        **fmm3dmain_kwargs)
else:
    hfmm3dmain(
        zk=np.complex128(zkfmm),
        radssort=radssort,
        jsort=texpssort,
        **fmm3dmain_kwargs)

# src/Laplace/lfmm3d.f#L501
if ifpgh >= 1:
    pot = reorder_inv(potsort, isrc)
if ifpgh >= 2:
    grad = reorder_inv(gradsort, isrc)
    grad *= b0inv
if ifpgh >= 3:
    hess = reorder_inv(hesssort, isrc)
    hess *= b0inv2

# src/Laplace/lfmm3d.f#L514
if ifpghtarg >= 1:
    pottarg = reorder_inv(pottargsort, itarg)
if ifpghtarg >= 2:
    gradtarg = reorder_inv(gradtargsort, itarg)
    gradtarg *= b0inv
if ifpghtarg >= 3:
    hesstarg = reorder_inv(hesstargsort, itarg)
    hesstarg *= b0inv2

pot2 = np.zeros(nsource, dtype=np.double)
for i in range(nsource):
    for j in range(nsource):
        if i == j:
            continue
        x = source[:, i]
        y = source[:, j]
        pot2[i] += charge[0, j]/np.linalg.norm(x - y)

print(pot2)
print(pot)
print(np.max(np.abs(pot2-pot)))
