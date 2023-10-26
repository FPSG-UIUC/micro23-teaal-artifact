import os
import sys
sys.path.insert(0, "..")

import scripts.extensor as extensor
import scripts.extensor_energy as extensor_energy
import scripts.gamma as gamma
import scripts.outerspace as outerspace
import scripts.sigma as sigma

def main(pregenerated):
    plot_dir = "../data/plots/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig = extensor.graph_mem(pregenerated=pregenerated)
    fig.savefig(plot_dir + "fig9a.pdf", bbox_inches='tight')
    fig = gamma.graph_mem(pregenerated=pregenerated)
    fig.savefig(plot_dir + "fig9b.pdf", bbox_inches='tight')
    fig = outerspace.graph_mem(pregenerated=pregenerated)
    fig.savefig(plot_dir + "fig9c.pdf", bbox_inches='tight')


    fig = extensor.graph_speedup(pregenerated=pregenerated)
    fig.savefig(plot_dir + "fig10a.pdf", bbox_inches='tight')
    fig = gamma.graph_speedup(pregenerated=pregenerated)
    fig.savefig(plot_dir + "fig10b.pdf", bbox_inches='tight')
    fig = outerspace.graph_time(pregenerated=pregenerated)
    fig.savefig(plot_dir + "fig10c.pdf", bbox_inches='tight')
    fig = sigma.graph_speedup(pregenerated=pregenerated)
    fig.savefig(plot_dir + "fig10d.pdf", bbox_inches='tight')

    fig = extensor_energy.graph_energy(pregenerated=pregenerated)
    fig.savefig(plot_dir + "fig11.pdf", bbox_inches='tight')

if __name__ == "__main__":
    if sys.argv[1] == "pregenerated":
        main(True)
    else:
        main(False)
