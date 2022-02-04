from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)
import pytest

import pyopencl as cl
from numpy.random import default_rng
from pytools.obj_array import make_obj_array
from pyopencl.clrandom import PhiloxGenerator
import numpy as np
from boxtree import TreeBuilder
from boxtree.traversal import FMMTraversalBuilder
from boxtree.fmm3d.fortran import pts_tree_build, pts_tree_mem
from boxtree.fmm3d.treeinfo import fmm3d_tree_build


def get_test_data(ndiv, ctx, nparticles):
    queue = cl.CommandQueue(ctx)

    dims = 3

    np_rng = default_rng(10)
    vals = np_rng.random((3, nparticles - 2), dtype=np.double)
    # vals = np_rng.dirichlet((10, 5, 3), (nparticles - 2)).T
    particles_np = [
        np.append(vals[0], [1, 0]),
        np.append(vals[1], [1, 0]),
        np.append(vals[2], [1, 0]),
    ]

    particles = make_obj_array([
        cl.array.to_device(queue, particles_np[i])
        for i in range(dims)])

    rng = PhiloxGenerator(ctx, seed=15)
    charge = rng.normal(queue, nparticles, dtype=np.float64).get(
            queue).reshape((1, nparticles))
    dipvec = np.asfortranarray(
            rng.normal(queue, (1, 3, nparticles), dtype=np.float64).get(queue))

    tb = TreeBuilder(ctx)
    device_tree, _ = tb(
        queue, particles, max_particles_in_box=ndiv, kind='adaptive',
        skip_prune=False,
        bbox=np.array([[0, 1], [0, 1], [0, 1]], dtype=np.double))

    tg = FMMTraversalBuilder(ctx)
    device_trav, _ = tg(queue, device_tree)

    trav = device_trav.get(queue)
    tree = trav.tree

    return tree, trav, charge, dipvec, particles_np


@pytest.mark.parametrize("nparticles", [500, 5000, 50000])
def test_treeinfo(ctx_factory, nparticles):
    ctx = ctx_factory()
    ndiv = 40
    tree, trav, charge, dipvec, particles_np = get_test_data(ndiv, ctx,
        nparticles)
    itree, ltree, ipointer, treecenters, boxsize, \
        source, nsource, targ, ntarg, expc, nexpc, \
        isrc, itarg, iexpc, isrcse, itargse, iexpcse, \
        nlevels, nboxes = fmm3d_tree_build(tree, trav)

    nlevels_ref = np.array([0], dtype=np.int32)
    nboxes_ref = np.array([0], dtype=np.int32)
    ltree_ref = np.array([0], dtype=np.int64)

    source = np.array(particles_np, dtype=np.double, order='F')

    pts_tree_mem(
        src=source,
        ns=nsource,
        targ=targ,
        nt=ntarg,
        idivflag=0,
        ndiv=ndiv,
        nlmin=0,
        nlmax=51,
        ifunif=0,
        iper=1,
        nlevels=nlevels_ref,
        nboxes=nboxes_ref,
        ltree=ltree_ref)

    # nboxes = nboxes_ref[0]
    # nlevels = nlevels_ref[0]
    # ltree = ltree_ref[0]
    assert nboxes == nboxes_ref[0]
    assert nlevels == nlevels_ref[0]
    assert ltree == ltree_ref[0]

    itree_ref = np.zeros(ltree, dtype=np.int32)
    iptr_ref = np.zeros(8, dtype=np.int64)
    treecenters_ref = np.zeros((3, nboxes), dtype=np.double, order='F')
    boxsize_ref = np.zeros(nlevels + 1, dtype=np.double)

    pts_tree_build(
        src=source,
        ns=nsource,
        targ=targ,
        nt=ntarg,
        idivflag=0,
        ndiv=ndiv,
        nlmin=0,
        nlmax=51,
        ifunif=0,
        iper=1,
        nlevels=nlevels,
        nboxes=nboxes,
        ltree=ltree,
        itree=itree_ref,
        iptr=iptr_ref,
        centers=treecenters_ref,
        boxsize=boxsize_ref)

    iptr = ipointer
    assert (itree == itree_ref).all()
    assert (iptr == iptr_ref).all()
    assert np.allclose(treecenters, treecenters_ref)
    assert np.allclose(boxsize, boxsize_ref)
