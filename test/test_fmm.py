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
from sumpy import P2P

from boxtree.fmm3d.fmm import (
    FMM3DExpansionWrangler, FMM3DTreeIndependentDataForWrangler)


@pytest.mark.parametrize("knl", (HelmholtzKernel(3), LaplaceKernel(3)))
def test_fmm(ctx_factory, knl):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    dims = 3
    nparticles = 1000000
    eps = 1e-5

    np_rng = default_rng(10)
    particles_host = np_rng.random((3, nparticles), dtype=np.double)

    particles = make_obj_array([
        cl.array.to_device(queue, particles_host[i])
        for i in range(dims)])

    rng = PhiloxGenerator(ctx, seed=15)
    charge_device = rng.normal(queue, nparticles, dtype=np.float64)
    charge = charge_device.get(queue).reshape((1, nparticles))
    # dipvec = np.asfortranarray(
    #        rng.normal(queue, (1, 3, nparticles),
    #                   dtype=np.float64).get(queue))

    ndiv = 1000
    tb = TreeBuilder(ctx)
    device_tree, _ = tb(
        queue, particles, max_particles_in_box=ndiv,
        kind='adaptive',
        bbox=np.array([[0, 1], [0, 1], [0, 1]], dtype=np.double))

    tg = FMMTraversalBuilder(ctx)
    device_trav, _ = tg(queue, device_tree)

    trav = device_trav.get(queue=queue)

    source_kernels = (knl,)
    target_kernels = (knl,)
    extra_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kwargs["k"] = 0.1

    data = FMM3DTreeIndependentDataForWrangler(
        ctx, target_kernels=target_kernels, source_kernels=source_kernels,
        strength_usage=(0,))

    wrangler = FMM3DExpansionWrangler(
        data, trav, source_extra_kwargs=extra_kwargs,
        kernel_extra_kwargs=extra_kwargs, eps=eps)

    extra_kwargs["target_to_source"] = np.arange(nparticles, dtype=np.int32)
    p2p = P2P(ctx, target_kernels=target_kernels,
              source_kernels=source_kernels,
              exclude_self=True)

    pot, = drive_fmm(wrangler, charge)
    _, (pot_ref,) = p2p(
            queue, particles, particles, (charge_device,),
            out_host=True, **extra_kwargs)

    assert np.max(np.abs(pot_ref - pot)) < eps
