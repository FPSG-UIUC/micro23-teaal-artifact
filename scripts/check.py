import os
import sys
sys.path.insert(0, "..")

from fibertree import Tensor

import scripts.extensor as extensor
import scripts.extensor_energy as extensor_energy
import scripts.gamma as gamma
import scripts.outerspace as outerspace
import scripts.sigma as sigma

def main():
    K = 8
    M = 6
    N = 7
    density = 0.5

    # Create the tensors
    A_KM = Tensor.fromRandom(rank_ids=["K", "M"], shape=[K, M], density=[0.9, density], seed=0)
    B_KN = Tensor.fromRandom(rank_ids=["K", "N"], shape=[K, N], density=[0.9, density], seed=1)

    # Run the accelerators
    extensor.check(extensor.run(A_KM, B_KN, 2, 2, 2, 2))

    extensor_energy.check(extensor_energy.run(A_KM, B_KN, 2, 2, 2, 2))

    A_MK = A_KM.swizzleRanks(["M", "K"])
    gamma.check(gamma.run(A_MK, B_KN))

    outerspace.check(outerspace.run(A_KM, B_KN))

    sigma.check(sigma.run(A_KM, B_KN))

if __name__ == "__main__":
    main()
