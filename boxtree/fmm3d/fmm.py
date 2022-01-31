from boxtree.fmm3d.treeinfo import fmm3d_tree_build
from boxtree.fmm3d.fortran import (l3dterms, h3dterms, mpalloc,
                                   lfmm3dmain, hfmm3dmain)
from boxtree.fmm import (ExpansionWranglerInterface,
    TreeIndependentDataForWrangler)
from sumpy.kernel import (LaplaceKernel, HelmholtzKernel,
    KernelWrapper, AxisSourceDerivative, DirectionalSourceDerivative)
from pytools import memoize_method
import numpy as np


def reorder(arr, iarr):
    return arr[..., iarr - 1]


def reorder_inv(arr, iarr):
    res = arr.copy()
    res[..., iarr - 1] = arr
    return res


class FMM3DTimingFuture:

    def __init__(self, time_taken=0):
        self.time_taken = time_taken

    @memoize_method
    def result(self):
        from boxtree.timing import TimingResult
        return TimingResult(wall_elapsed=self.time_taken)

    def done(self):
        return True


class FMM3DExpansionWrangler(ExpansionWranglerInterface):
    """Implements the :class:`boxtree.fmm.ExpansionWranglerInterface`
    by using FMM3D expansions/translations.

    .. attribute:: source_extra_kwargs
        Keyword arguments to be passed to interactions that involve
        source particles.

    .. attribute:: kernel_extra_kwargs
        Keyword arguments to be passed to interactions that involve
        expansions, but not source particles.
    """
    def __init__(self, tree_indep, traversal, source_extra_kwargs,
            kernel_extra_kwargs, eps):
        super().__init__(tree_indep, traversal)
        self.source_extra_kwargs = source_extra_kwargs
        self.kernel_extra_kwargs = kernel_extra_kwargs
        self.eps = eps

        with cl.CommandQueue(tree_indep.cl_context) as queue:
            self.dipole_vecs = {
                name: np.array([d_i.get(queue=queue)
                        for d_i in source_extra_kwargs[name]],
                        order="F") for name in tree_indep.dipole_vec_names}

        if tree_indep.is_helmholtz:
            self.zk = kernel_extra_kwargs[
                    tree_indep.base_kernel.helmholtz_k_name]
        else:
            self.zk = 0


    def reorder_potential(self, potential):
        return potential

    def reorder_sources(self, source_array):
        return source_array.get(source_array.queue)[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        import numpy as np
        assert (
                isinstance(potentials, np.ndarray)
                and potentials.dtype.char == "O")

        def reorder(x):
            return x[self.tree.sorted_target_ids]

        return np.array(map(reorder, potentials), dtype=object)

    def finalize_potentials(self, potentials, template_ary):
        return potentials

    def form_locals(self, *args):
        return 0, FMM3DTimingFuture(0)

    def form_multipoles(self, *args):
        return 0, FMM3DTimingFuture(0)

    def coarsen_multipoles(self, *args):
        return 0, FMM3DTimingFuture(0)

    def multipole_to_local(self, *args):
        return 0, FMM3DTimingFuture(0)

    def refine_locals(self, *args):
        return 0, FMM3DTimingFuture(0)

    def eval_locals(self, *args):
        return 0, FMM3DTimingFuture(0)

    def eval_multipoles(self, *args):
        return 0, FMM3DTimingFuture(0)

    def eval_direct(self,
            target_boxes, neighbor_sources_starts,
            neighbor_sources_lists, src_weight_vecs):

        strength_usage = self.tree_indep.strength_usage

        ifcharge = 0
        ifdipole = 1
        charge = None
        dipvec = [0, 0, 0]

        for i, kernel in enumerate(self.tree_indep.source_kernels):
            strength =  src_weight_vecs[strength_usage[i]]
            if isinstance(kernel, KernelWrapper):
                ifdipole = 1
                if isinstance(kernel, AxisSourceDerivative):
                    dipvec[kernel.axis] += strength
                elif isinstance(kernel, DirectionalSourceDerivative):
                    for d in range(3):
                        dipvec[d] += strength * \
                                self.dipole_vecs[kernel.dir_vec_name]
            else:
                ifcharge = 1
                charge = strength

        nsource = len(self.tree.sources[0])

        if ifdipole:
            for i in range(3):
                if not isinstance(dipvec[i], np.ndarray):
                    dipvec[i] = np.zeros(nsource, dtype=np.double)
            dipvec = np.array(dipvec, order='F')

        if self.tree.sources_are_targets:
            ifpghtarg = 0
            ifpgh = self.tree.target_deriv_count + 1
        else:
            ifpghtarg = self.tree.target_deriv_count + 1
            ifpgh = 0

        pot, grad, hess, pottarg, gradtarg, hesstarg = \
            _run_fmm(self.tree, self.traversal, charge, dipvec,
                    ifdipole, ifcharge, ifpgh, ifpghtarg, self.zk, self.eps)

        if not self.tree.source_are_targets:
            pot, grad, hess = pottarg, gradtard, hesstarg

        result = []
        for kernel in self.target_kernels:
            if not isinstance(kernel, KernelWrapper):
                result.append(pot)
            else:
                raise NotImplementedError("Not implemented yet")

        return result, FMM3DTimingFuture(0)


class FMM3DTreeIndependentDataForWrangler(TreeIndependentDataForWrangler):
    def __init__(self, cl_context,
            target_kernels, source_kernels, strength_usage):
        self.cl_context = cl_context
        self.target_kernels = target_kernels
        self.source_kernels = source_kernels
        self.strength_usage = strength_usage

        self.base_kernel = self.source_kernels[0].get_base_kernel()
        self.is_helmholtz = isinstance(self.base_kernel, HelmholtzKernel)

        self.dipole_vec_names = []

        for kernel in self.source_kernels:
            if not isinstance(kernel.get_base_kernel(),
                              (LaplaceKernel, HelmholtzKernel)):
                raise ValueError("Only Laplace and Helmholtz allowed")
            if isinstance(kernel, KernelWrapper):
                if isinstance(kernel.inner_kernel, KernelWrapper):
                    raise ValueError(
                            "Only one source derivative transformation allowed.")
                if not isinstance(kernel,
                        (AxisSourceDerivative, DirectionalSourceDerivative)):
                    raise ValueError(
                            "Only axis and directional source derivatives allowed")
                if isinstance(kernel, DirectionalSourceDerivative):
                    self.dipole_vec_names.append(kernel.dir_vec_name)

        self.target_deriv_count = 0
        for kernel in self.target_kernels:
            deriv_count = 0
            inner_kernel = kernel
            while isinstance(inner_kernel, KernelWrapper):
                if isinstance(inner_kernel,
                        (AxisTargetDerivative, DirectionalTargetDerivative)):
                    deriv_count += 1
                inner_kernel = kernel.inner_kernel

            if isinstance(inner_kernel, LaplaceKernel) and deriv_count > 2:
                raise ValueError("Cannot take more than two target "
                    "derivatives for Laplace kernel")

            if isinstance(inner_kernel, HelmholtzKernel) and deriv_count > 1:
                raise ValueError("Cannot take more than one target "
                    "derivative for Helmholtz kernel")

            self.target_deriv_count = max(self.target_deriv_count, deriv_count)


def _run_fmm(tree, trav, charge, dipvec, ifdipole, ifcharge, ifpgh, ifpghtarg, zk, eps):
    # number of fmms
    nd = 1
    # flag for periodic implmentations. Currently unused
    iper = 0
    ifnear = 1

    itree, ltree, ipointer, treecenters, boxsize, \
        source, nsource, targ, ntarg, expc, nexpc, \
        isrc, itarg, iexpc, isrcse, itargse, iexpcse, \
        nlevels, nboxes = fmm3d_tree_build(tree, trav)

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
        ndiv=np.int32(0),  # not used
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

    pot = None
    grad = None
    hess = None
    pottarg = None
    gradtarg = None
    hesstarg = None

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

    return pot, grad, hess, pottarg, gradtarg, hesstarg


if __name__ == "__main__":
    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    dims = 3
    nparticles = 500

    from numpy.random import default_rng
    np_rng = default_rng(10)
    vals = np_rng.random((3, nparticles - 2), dtype=np.double)
    #vals = np_rng.dirichlet((10, 5, 2), (nparticles - 2)).T
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
    device_tree, _ = tb(queue, particles, max_particles_in_box=ndiv, kind='adaptive',
                 bbox=np.array([[0, 1], [0, 1], [0, 1]], dtype=np.double))

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(ctx)
    device_trav, _ = tg(queue, device_tree)

    trav = device_trav.get(queue=queue)
    tree = trav.tree

    pot, grad, hess, pottarg, gradtarg, hesstarg = \
        _run_fmm(tree, trav, charge, dipvec,
            ifdipole=0, ifcharge=1, ifpgh=1, ifpghtarg=0, zk=0, eps=1e-5)

    source = np.array([row for row in tree.sources])

    pot2 = np.zeros(nparticles, dtype=np.double)
    for i in range(nparticles):
        for j in range(nparticles):
            if i == j:
                continue
            x = source[:, i]
            y = source[:, j]
            pot2[i] += charge[0, j]/np.linalg.norm(x - y)

    print(np.max(np.abs(pot2-pot)))

