# Load the tensors as fibertrees

import scipy.io as sio

from fibertree import Tensor


def getSciPy(name):
    loaded = sio.loadmat("tensors/" + name + ".mat")

    # Get the actual location of the matrix
    ind_0 = ["wiki-Vote-gp", "p2p-Gnutella31-gp", "ca-CondMat-gp",
             "poisson3Da-gp", "email-Enron-gp"]
    ind_1 = ["poisson3Da"]
    ind_2 = ["wiki-Vote", "p2p-Gnutella31", "ca-CondMat", "email-Enron",
             "flickr", "wikipedia-20070206", "soc-LiveJournal1"]

    if name in ind_0:
        i = 0
    elif name in ind_1:
        i = 1
    elif name in ind_2:
        i = 2
    else:
        print(loaded["Problem"][0][0])
        raise ValueError

    return loaded["Problem"][0][0][i]


def convert(name):
    """
    Convert from a scipy matrix to an HFA tensor
    """
    mat = getSciPy(name)

    # Build the HFA Tensor
    A_MK = Tensor(rank_ids=["M", "K"], shape=mat.get_shape())
    coo = mat.tocoo()

    a_m = A_MK.getRoot()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        a_ref = a_m.getPayloadRef(i, j)
        a_ref += v

    print("Loaded", name)

    return A_MK

if __name__ == "__main__":
    convert("wiki-Vote")
    convert("p2p-Gnutella31")
    convert("ca-CondMat")
    convert("poisson3Da")
    convert("email-Enron")
