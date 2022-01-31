import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import pyopencl as cl
import numpy as np
from numpy.random import default_rng
from pytools.obj_array import make_obj_array
from pyopencl.clrandom import PhiloxGenerator
from boxtree.traversal import FMMTraversalBuilder
from boxtree import TreeBuilder
from boxtree.fmm import drive_fmm
from sumpy.kernel import LaplaceKernel, HelmholtzKernel

from boxtree.fmm3d.fmm import FMM3DExpansionWrangler, FMM3DTreeIndependentDataForWrangler 


def test_fmm(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    
    dims = 3
    nparticles = 500
    eps = 1e-5
    
    np_rng = default_rng(10)
    particles_host = np_rng.random((3, nparticles), dtype=np.double)
    
    particles = make_obj_array([
        cl.array.to_device(queue, particles_host[i])
        for i in range(dims)])
    
    rng = PhiloxGenerator(ctx, seed=15)
    charge = rng.normal(queue, nparticles, dtype=np.float64).get(
            queue).reshape((1, nparticles))
    dipvec = np.asfortranarray(
            rng.normal(queue, (1, 3, nparticles), dtype=np.float64).get(queue))
    
    ndiv = 40
    tb = TreeBuilder(ctx)
    device_tree, _ = tb(queue, particles, max_particles_in_box=ndiv, kind='adaptive',
                 bbox=np.array([[0, 1], [0, 1], [0, 1]], dtype=np.double))
    
    tg = FMMTraversalBuilder(ctx)
    device_trav, _ = tg(queue, device_tree)
    
    trav = device_trav.get(queue=queue)
    tree = trav.tree
    
    knl = LaplaceKernel(3)
    
    data = FMM3DTreeIndependentDataForWrangler(ctx,
        target_kernels=(knl,), source_kernels=(knl,), strength_usage=(0,))
    wrangler = FMM3DExpansionWrangler(data, trav, source_extra_kwargs={},
                kernel_extra_kwargs={}, eps=eps)
    
    pot = drive_fmm(wrangler, charge)
    
    source = np.array([row for row in tree.sources])
    
    pot_ref = np.zeros(nparticles, dtype=np.double)
    for i in range(nparticles):
        for j in range(nparticles):
            if i == j:
                continue
            x = particles_host[:, i]
            y = particles_host[:, j]
            pot_ref[i] += charge[0, j]/np.linalg.norm(x - y)
    
    assert np.max(np.abs(pot_ref - pot)) < eps
    
