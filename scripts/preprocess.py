import functools
import heapq
import os

import scipy.io as sio
from scipy.sparse import coo_array

from scripts.convert import getSciPy

@functools.total_ordering
class QueueElem:
    """An element to be held in the priority queue"""
    def __init__(self, row):
        self.row = row
        self.key = 0

    def incKey(self):
        self.key += 1

    def decKey(self):
        self.key -= 1

    def __eq__(self, other):
        return self.key == other.key

    def __lt__(self, other):
        # For turning a minheap into a maxheap
        return self.key > other.key

    def __repr__(self):
        return "(" + str(self.row) + ", " + str(self.key) + ")"


def preprocess_one(old_mat, cache_sz):
    """
    Perform Gamma preprocessing
    """
    coo = old_mat.tocoo()
    csr_dict = {}
    csc_dict = {}

    # Build up a CSR and CSC version of this matrix
    for row, col, val in zip(coo.row, coo.col, coo.data):
        if row not in csr_dict:
            csr_dict[row] = []
        csr_dict[row].append((col, val))

        if col not in csc_dict:
            csc_dict[col] = []
        csc_dict[col].append(row)

    # Implement Algorithm 1 in the Gamma paper
    heap = []
    elems = {}
    for row in csr_dict.keys():
        elems[row] = QueueElem(row)
        heap.append(elems[row])
    heapq.heapify(heap)

    # 12 bytes = 4 byte coords + 8 byte payloads
    max_nnzs_fibercache = cache_sz / 12
    # Note: Because the matrices are all square and A and B are just transposes
    # of each other, nnz / row of A and nnz / row of B are the same
    nnz_row = coo.getnnz() / coo.get_shape()[0]
    W = int(max_nnzs_fibercache / (nnz_row * nnz_row))

    perm = []
    i = 0

    perm.append(heapq.heappop(heap))
    del elems[perm[i].row]
    while heap:
        # Incremement key of matching rows
        for col, _ in csr_dict[perm[i].row]:
            for row in csc_dict[col]:
                if row in elems:
                    elems[row].incKey()

        # Decrement key of those elements outside the exploitable reuse distance
        if i >= W:
            for col, _ in csr_dict[perm[i - W - 1].row]:
                for row in csc_dict[col]:
                    if row in elems:
                        elems[row].decKey()

        heapq.heapify(heap)
        perm.append(heapq.heappop(heap))

        i += 1
        del elems[perm[i].row]

    rows = []
    cols = []
    data = []
    for i, elem in enumerate(perm):
        for col, val in csr_dict[elem.row]:
            rows.append(i)
            cols.append(col)
            data.append(val)

    new_mat = coo_array((data, (rows, cols)), shape=old_mat.shape)
    return new_mat

def gamma():
    mats = ["wiki-Vote", "p2p-Gnutella31", "ca-CondMat", "poisson3Da",
            "email-Enron"]

    for mat in mats:
        if os.path.exists("tmp/tensors/" + mat + "-gp.mat"):
            return

        old_mat = getSciPy(mat)
        new_mat = preprocess_one(old_mat, 3 * 2**20)
        mdict = {"Problem": [[[new_mat]]]}
        sio.savemat("tmp/tensors/" + mat + "-gp.mat", mdict)

