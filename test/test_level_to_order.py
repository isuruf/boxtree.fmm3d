import pytest
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from boxtree.fmm3d.level_to_order import FMM3DExpansionOrderFinder
from sumpy.kernel import LaplaceKernel, HelmholtzKernel
import numpy as np

# Code from inducer/sumpy
# Copyright (C) 2017 Andreas Kloeckner
# SPDX-License-Identifier: MIT
class FakeTree:
    def __init__(self, dimensions, root_extent, stick_out_factor):
        self.dimensions = dimensions
        self.root_extent = root_extent
        self.stick_out_factor = stick_out_factor

@pytest.mark.parametrize("knl", (LaplaceKernel(3),))
def test_level_to_order(ctx_factory, knl):
    ofind = FMM3DExpansionOrderFinder(1e-5)

    tree = FakeTree(knl.dim, 1, 0.5)
    orders = [
        ofind(knl, frozenset([("k", 0.1)]), tree, level)
        for level in range(10)]
    print(orders)

    if isinstance(knl, HelmholtzKernel):
        pytest.xfail("Helmholtz h3dterms fails, but it should not.")

    # Order should not increase with level
    assert (np.diff(orders) <= 0).all()
