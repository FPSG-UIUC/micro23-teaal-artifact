import os
import sys
sys.path.insert(0, "..")

import scripts.extensor as extensor
import scripts.extensor_energy as extensor_energy
import scripts.gamma as gamma
import scripts.outerspace as outerspace
import scripts.sigma as sigma

def main():
    extensor.eval()
    extensor.run_sparseloop()
    gamma.eval()
    outerspace.eval_mem()
    outerspace.eval_time()
    sigma.eval()
    extensor_energy.eval(dataset="all")

if __name__ == "__main__":
    main()
