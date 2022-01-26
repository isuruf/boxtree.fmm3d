import numpy as np
from boxtree.fmm3d.fortran import pts_tree_sort, pts_tree_build, pts_tree_mem

def fmm3d_tree_build(tree, trav, queue):
    # src/Laplace/lfmm3d.f L213-L240
    nlevels = tree.nlevels - 1
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

    for i in range(nlevels + 1):
        # first box for ith level
        itree[2*i] = tree.level_start_box_nrs[i] + 1
        # last box for ith level
        itree[2*i + 1] = tree.level_start_box_nrs[i + 1]

    # level of the boxes
    itree[iptr[1] - 1:iptr[2] - 1] = box_levels

    # parent of the boxes
    itree[iptr[2] - 1:iptr[3] - 1] = box_parent_ids + 1
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
    itree[iptr[5] - 1:iptr[6] - 1] = coll_starts[1:] - coll_starts[:-1] + 1

    # icolleagues
    itree[iptr[6] - 1:iptr[7] - 1] = -1
    for i in range(nboxes):
        istart = iptr[6] + 27*i - 1
        colls = coll_lists[coll_starts[i]:coll_starts[i+1]] + 1
        colls = np.sort(np.append(colls, [i + 1]))
        itree[istart:istart+itree[iptr[5] + i - 1]] = colls

    boxsize = np.zeros(nlevels + 1, dtype=np.double)
    boxsize[0] = tree.root_extent
    for i in range(nlevels):
        boxsize[i + 1] = boxsize[i] / 2

    nexpc = 0
    expc = np.zeros((3, nexpc), dtype=np.double, order='F')

    source = np.array([row.get(queue) for row in tree.sources], order='F')
    nsource = source.shape[1]

    targ = np.array([row.get(queue) for row in tree.targets], order='F')
    ntarg = targ.shape[1]

    targ = np.zeros((3, 0), order='F')
    ntarg = 0
    
    treecenters = np.asfortranarray(box_centers)

    print("nlevels before", nlevels)
    print("nboxes before", nboxes)
    nlevels = np.array([0], dtype=np.int32)
    nboxes = np.array([0], dtype=np.int32)
    ltree = np.array([0], dtype=np.int64)
    pts_tree_mem(src=source,ns=nsource,targ=targ,nt=ntarg,idivflag=0,ndiv=40,nlmin=0,nlmax=51,
        ifunif=0,iper=1,nlevels=nlevels,nboxes=nboxes,ltree=ltree)
    nboxes = int(nboxes[0])
    nlevels = int(nlevels[0])
    ltree = int(ltree[0])

    print("nlevels after", nlevels)
    print("nboxes after", nboxes)
    boxsize = np.zeros(nlevels + 1, dtype=np.double)
    boxsize[0] = tree.root_extent
    for i in range(nlevels):
        boxsize[i + 1] = boxsize[i] / 2

    print(nlevels)
    print(boxsize)
    print(iptr)
    print(list(itree))
    print("treecenters", treecenters)
    itree = np.zeros(ltree, dtype=np.int32)
    treecenters = np.asfortranarray(np.zeros((3, nboxes), dtype=np.double))
    pts_tree_build(src=source, ns=nsource, targ=targ, nt=ntarg, idivflag=0,
            ndiv=40, nlmin=0, nlmax=51, ifunif=0,
            iper=1, nlevels=nlevels, nboxes=nboxes, ltree=ltree,
            itree=itree, iptr=iptr, centers=treecenters, boxsize=boxsize)
    print(boxsize)
    print(iptr)
    print(list(itree))
    print("treecenters2", treecenters)
    print(ltree)
    print(nboxes)
    print(nlevels)

    isrc = np.zeros(nsource, dtype=np.int32)
    itarg = np.zeros(ntarg, dtype=np.int32)
    iexpc = np.zeros(1, dtype=np.int32)

    isrcse = np.zeros((2, nboxes), dtype=np.int32, order='F')
    itargse = np.zeros((2, nboxes), dtype=np.int32, order='F')
    iexpcse = np.zeros((2, nboxes), dtype=np.int32, order='F')

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

    if ntarg > 0:
        pts_tree_sort(n=ntarg, xys=targ, ixy=itarg, ixyse=itargse,
            **pts_tree_sort_kwargs)

    pts_tree_sort(n=nexpc, xys=expc, ixy=iexpc, ixyse=iexpcse,
        **pts_tree_sort_kwargs)

    print(isrc)
    print(isrcse)
    print(iexpc)
    print(iexpcse)

    return itree, ltree, iptr, treecenters, boxsize, \
        source, nsource, targ, ntarg, expc, nexpc, \
        isrc, itarg, iexpc, isrcse, itargse, iexpcse, \
        nlevels, nboxes
