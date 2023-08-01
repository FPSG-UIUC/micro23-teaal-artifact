# Download the data
import os

def download():
    """
    Download the data

    Should be run from inside /path/to/micro23-teaal-artifact/
    """
    if not os.path.exists(os.getcwd() + "/tensors"):
        os.makedirs(os.getcwd() + "/tensors")

    tensors = [
        "https://suitesparse-collection-website.herokuapp.com/mat/SNAP/wiki-Vote.mat",
        "https://suitesparse-collection-website.herokuapp.com/mat/SNAP/p2p-Gnutella31.mat",
        "https://suitesparse-collection-website.herokuapp.com/mat/SNAP/ca-CondMat.mat",
        "https://suitesparse-collection-website.herokuapp.com/mat/FEMLAB/poisson3Da.mat",
        "https://suitesparse-collection-website.herokuapp.com/mat/SNAP/email-Enron.mat"
    ]
    for tensor in tensors:
        os.system("wget -P tensors " + tensor)

if __name__ == "__main__":
    download()

