import numpy as np
from boxtree.fmm3d.fortran import pts_tree_sort, pts_tree_build, pts_tree_mem
from collections import deque

_child_mapping = [1, 5, 3, 7, 2, 6, 4, 8]
_child_mapping_inv = [0, 4, 2, 6, 1, 5, 3, 7]


def _get_child_number(dirs):
    # https://github.com/flatironinstitute/FMM3D/blob/6e3add2a8fb7cfc2b6b723975ff36f2adea48e28/src/Common/pts_tree3d.f#L727-L734
    num = int("".join([str(int(dir)) for dir in dirs]), 2)
    return _child_mapping[num]


def _get_child_dir(num):
    bin_rep = format(_child_mapping_inv[num - 1], "03b")
    return np.array([1.0 if b == "1" else -1.0 for b in bin_rep],
                    dtype=np.double)


def fmm3d_tree_build(tree, trav):
    # src/Laplace/lfmm3d.f L213-L240
    nlevels = tree.nlevels - 1
    nboxes_pruned = tree.nboxes

    box_levels_pruned = tree.box_levels
    box_parent_ids_pruned = tree.box_parent_ids
    box_child_ids_pruned = tree.box_child_ids[:, :nboxes_pruned]
    coll_starts_pruned = trav.colleagues_starts
    coll_lists_pruned = trav.colleagues_lists
    box_centers_pruned = tree.box_centers[:, :nboxes_pruned]
    level_start_box_nrs_pruned = tree.level_start_box_nrs

    box_id_mapping = np.zeros_like(box_parent_ids_pruned)

    boxsize = np.zeros(nlevels + 1, dtype=np.double)
    boxsize[0] = tree.root_extent
    for i in range(nlevels):
        boxsize[i + 1] = boxsize[i] / 2

    new_box_counter = 0
    box_id_mapping[0] = 0
    level_start_box_nrs = np.ones(nlevels + 2, dtype=np.int32) * -1

    box_levels = [0]
    box_parent_ids = [0]
    box_child_ids = [[0]*8]
    box_centers = [box_centers_pruned[:, 0]]
    id_to_pruned_id_mapping = [0]
    pruned_id_to_id_mapping = np.zeros(nboxes_pruned, dtype=np.int32)

    box_id = 0
    while box_id <= new_box_counter:
        level = box_levels[box_id]
        center = box_centers[box_id]
        child_nums = []
        have_children = False
        pruned_id = id_to_pruned_id_mapping[box_id]

        if pruned_id != -1:
            # if this box is not in the prune tree, it doesn't have children
            for child_id in box_child_ids_pruned[:, pruned_id]:
                if child_id != 0:
                    center_child = box_centers_pruned[:, child_id]
                    dirs = [center[j] - center_child[j] <= 0 for j in range(3)]
                    child_num = _get_child_number(dirs)
                    child_nums.append(child_num)
                    new_child_id = new_box_counter + child_num
                    if len(id_to_pruned_id_mapping) <= new_child_id:
                        id_to_pruned_id_mapping.extend(
                            [-1]*(new_child_id + 1 -
                                  len(id_to_pruned_id_mapping)))
                    id_to_pruned_id_mapping[new_child_id] = child_id
                    pruned_id_to_id_mapping[child_id] = new_child_id
                    have_children = True

        if have_children:
            box_levels.extend([level + 1]*8)
            box_parent_ids.extend([box_id]*8)
            for j in range(8):
                child_center = center + \
                        _get_child_dir(j + 1) * boxsize[level + 1]/2
                child_id = new_box_counter + j + 1
                box_centers.append(child_center)
                box_child_ids.append([0]*8)
            box_child_ids[box_id] = list(range(new_box_counter + 1,
                                         new_box_counter + 9))
            new_box_counter += 8
            if len(id_to_pruned_id_mapping) <= new_box_counter:
                id_to_pruned_id_mapping.extend(
                    [-1]*(new_box_counter + 1 - len(id_to_pruned_id_mapping)))

        if level_start_box_nrs[level] == -1:
            level_start_box_nrs[level] = box_id

        box_id += 1

    nboxes = new_box_counter + 1
    level_start_box_nrs[nlevels + 1] = nboxes

    box_centers = np.asfortranarray(np.array(box_centers).T)
    box_child_ids = np.asfortranarray(np.array(box_child_ids).T)
    box_parent_ids = np.array(box_parent_ids)
    box_levels = np.array(box_levels)

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
        itree[2*i] = level_start_box_nrs[i] + 1
        # last box for ith level
        itree[2*i + 1] = level_start_box_nrs[i + 1]

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
        itree[istart:istart + len(non_zero_child_boxes)] = \
            non_zero_child_boxes + 1

    # ncolleagues
    itree[iptr[5] - 1:iptr[6] - 1] = 0
    # icolleagues
    itree[iptr[6] - 1:iptr[7] - 1] = -1
    if 0:
        for i in range(nboxes):
            pruned_id = id_to_pruned_id_mapping[i]
            if pruned_id != -1:
                coll_start = coll_starts_pruned[pruned_id]
                coll_end = coll_starts_pruned[pruned_id + 1]
                colls = pruned_id_to_id_mapping[
                        coll_lists_pruned[coll_start:coll_end]] + 1
                colls = np.sort(np.append(colls, [i + 1]))
            else:
                colls = []
            istart = iptr[6] + 27*i - 1
            itree[istart:istart+len(colls)] = colls
            itree[iptr[5] - 1 + i] = len(colls)
    else:
        icolleagues, ncolleagues = compute_colleagues(
                nboxes, box_parent_ids, box_child_ids,
                box_centers, box_levels, boxsize)
        itree[iptr[5] - 1:iptr[6] - 1] = ncolleagues
        itree[iptr[6] - 1:iptr[7] - 1] = icolleagues.reshape(-1) + 1

    nexpc = 0
    expc = np.zeros((3, nexpc), dtype=np.double, order='F')

    source = np.array([row for row in tree.sources], order='F')
    nsource = source.shape[1]

    targ = np.array([row for row in tree.targets], order='F')
    ntarg = targ.shape[1]

    if tree.sources_are_targets:
        targ = np.zeros((3, 0), order='F')
        ntarg = 0

    treecenters = np.asfortranarray(box_centers)
    """
    nlevels_ref = np.array([0], dtype=np.int32)
    nboxes_ref = np.array([0], dtype=np.int32)
    ltree_ref = np.array([0], dtype=np.int64)

    ndiv=40

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

    print(nboxes, nboxes_ref[0])
    nboxes = nboxes_ref[0]
    nlevels = nlevels_ref[0]
    ltree = ltree_ref[0]
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

    print(itree[iptr[5] - 1:iptr[6] - 1])
    print(itree_ref[iptr_ref[5] - 1:iptr_ref[6] - 1])

    treecenters = treecenters_ref
    boxsize = boxsize_ref
    itree = itree_ref
    iptr = iptr_ref
    """

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

    isrc, isrcse = pts_tree_sort2(nsource, tree, trav, id_to_pruned_id_mapping,
                                  nboxes, box_child_ids)

    if ntarg > 0:
        pts_tree_sort(n=ntarg, xys=targ, ixy=itarg, ixyse=itargse,
                      **pts_tree_sort_kwargs)

    pts_tree_sort(n=nexpc, xys=expc, ixy=iexpc, ixyse=iexpcse,
                  **pts_tree_sort_kwargs)

    return itree, ltree, iptr, treecenters, boxsize, \
        source, nsource, targ, ntarg, expc, nexpc, \
        isrc, itarg, iexpc, isrcse, itargse, iexpcse, \
        nlevels, nboxes


def compute_colleagues(nboxes, box_parent_ids, box_child_ids, box_centers,
                       box_levels, boxsizes):
    icolleagues = np.ones((nboxes, 27), dtype=np.int32) * -2
    ncolleagues = np.ones(nboxes, dtype=np.int32)

    icolleagues[0, 0] = 0

    if nboxes >= 1:
        icolleagues[1:9, :8] = np.arange(1, 9)
        ncolleagues[1:9] = 8

    for box_id in range(9, nboxes):
        parent_id = box_parent_ids[box_id]
        grandparent_id = box_parent_ids[parent_id]
        center = box_centers[:, box_id]
        level = box_levels[box_id]
        boxsize = boxsizes[level]
        count_coll = 0
        for j in range(8):
            parent_sibling_id = box_child_ids[j, grandparent_id]
            if parent_sibling_id == 0:
                continue
            for k in range(8):
                cousin_id = box_child_ids[k, parent_sibling_id]
                if cousin_id == 0:
                    continue
                cousin_center = box_centers[:, cousin_id]
                diff = np.abs(center - cousin_center)
                if (diff < 1.05 * boxsize).all():
                    icolleagues[box_id, count_coll] = cousin_id
                    count_coll += 1
        ncolleagues[box_id] = count_coll
    return icolleagues, ncolleagues


def pts_tree_sort2(n, tree, trav, id_to_pruned_id_mapping, nboxes,
                   box_child_ids):
    ixy = np.zeros(n, dtype=np.int32)
    ixyse = np.zeros((2, nboxes), dtype=np.int32)
    box_depth_first_ordering = np.zeros(nboxes, dtype=np.int32) * (-1)

    boxes_stack = deque()
    boxes_stack.append(0)

    i = 0
    while len(boxes_stack) > 0:
        current_box = boxes_stack.pop()
        box_depth_first_ordering[i] = current_box
        i = i + 1
        children = box_child_ids[:, current_box]
        for child_id in reversed(children):
            if child_id != 0:
                boxes_stack.append(child_id)

    box_source_starts = tree.box_source_starts
    box_source_counts_cumul = tree.box_source_counts_cumul
    box_source_counts_nonchild = tree.box_source_counts_nonchild

    count = 1
    for box_id in box_depth_first_ordering:
        box_pruned_id = id_to_pruned_id_mapping[box_id]
        if box_pruned_id == -1:
            ixyse[:, box_id] = [count + 1, count]
        else:
            npoints = box_source_counts_cumul[box_pruned_id]
            ixyse[:, box_id] = [count, count + npoints - 1]
            npoints_nonchild = box_source_counts_nonchild[box_pruned_id]
            point_starts = box_source_starts[box_pruned_id]
            if npoints_nonchild != 0:
                ixy[count - 1: count - 1 + npoints_nonchild] = \
                        np.arange(point_starts + 1,
                                  point_starts + 1 + npoints_nonchild)
            count += npoints_nonchild

    return ixy, ixyse
