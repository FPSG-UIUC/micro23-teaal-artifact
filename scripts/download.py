# Download the data
import os

def download():
    """
    Download the data

    Should be run from inside /path/to/micro23-teaal-artifact/
    """
    if not os.path.exists("tmp/tensors"):
        os.makedirs("tmp/tensors")

    tensors = {
        "wiki-Vote": "https://suitesparse-collection-website.herokuapp.com/mat/SNAP/wiki-Vote.mat",
        "p2p-Gnutella31": "https://suitesparse-collection-website.herokuapp.com/mat/SNAP/p2p-Gnutella31.mat",
        "ca-CondMat": "https://suitesparse-collection-website.herokuapp.com/mat/SNAP/ca-CondMat.mat",
        "poisson3Da": "https://suitesparse-collection-website.herokuapp.com/mat/FEMLAB/poisson3Da.mat",
        "email-Enron": "https://suitesparse-collection-website.herokuapp.com/mat/SNAP/email-Enron.mat"
    }
    for tensor in tensors:
        if not os.path.exists("tmp/tensors/" + tensor + ".mat"):
            os.system("wget -P tmp/tensors " + tensors[tensor])

if __name__ == "__main__":
    download()

