from boxtree.fmm3d.fortran import l3dterms, h3dterms
from sumpy.kernel import LaplaceKernel, HelmholtzKernel
import numpy as np


class FMM3DExpansionOrderFinder:

    def __init__(self, eps, extra_order=0):
        self.eps = eps
        self.extra_order = extra_order

    def __call__(self, kernel, kernel_args, tree, level):

        nterms = np.empty(1, dtype=np.int32)

        if isinstance(kernel, LaplaceKernel):
            # src/Laplace/lfmm3d.f#L392
            l3dterms(self.eps, nterms[0:1])

        elif isinstance(kernel, HelmholtzKernel):
            b0 = tree.root_extent
            zk = dict(kernel_args)[kernel.helmholtz_k_name]
            zkfmm = np.complex128(zk * b0)
            boxsize = 1.0 / 2 ** level
            h3dterms(boxsize, zkfmm, self.eps, nterms[0:1])

        return nterms[0] + self.extra_order
