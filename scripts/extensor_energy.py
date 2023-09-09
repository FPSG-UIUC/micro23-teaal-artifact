import os

import pandas as pd
from ruamel.yaml import YAML

from fibertree import Metrics, Tensor
from fibertree.model import Compute, Format, Traffic
from fibertree.model.intersect import SkipAheadIntersector

from teaal.parse import Einsum as EinsumParser
from teaal.parse import Mapping as MappingParser
from teaal.parse import Architecture as ArchitectureParser
from teaal.parse import Bindings as BindingsParser
from teaal.parse import Format as FormatParser
from teaal.trans.hifiber import HiFiber

from scripts.convert import convert
from scripts.download import download
import scripts.graph_utils as utils

def run(A_KM, B_KN, M1, K1, N1, pe_sz):
    assert A_KM.getRankIds() == ["K", "M"]
    assert B_KN.getRankIds() == ["K", "N"]

    M1 = M1 * pe_sz
    N1 = N1 * pe_sz
    K1 = K1 * pe_sz
    M0 = pe_sz
    N0 = pe_sz
    K0 = pe_sz

    K, M = A_KM.getShape()
    K, N = B_KN.getShape()

    fname = "../yamls/teaal/extensor-energy.yaml"
    einsum = EinsumParser.from_file(fname)
    mapping = MappingParser.from_file(fname)
    arch = ArchitectureParser.from_file(fname)
    bindings = BindingsParser.from_file(fname)
    format_ = FormatParser.from_file(fname)

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    exec(str(HiFiber(einsum, mapping, arch, bindings, format_)), globals(), locals())

    return locals()["metrics"]

def check(metrics):
    corr_metrics = {'Z': {'MainMemory': {'A': {'read': 7296}, 'B': {'read': 6464}, 'Z': {'read': 2560, 'write': 6912}}, 'LLB': {'A': {'read': 10560}, 'B': {'read': 12352}, 'Z': {'read': 2560, 'write': 6912}}, 'FPMul': {'mul': 46}, 'FPAdd': {'add': 13}, 'K2Intersect': 4, 'K1Intersect': 39, 'K0Intersection': 84, 'iter': {'N2': 2, 'K2': 4, 'M2': 8, 'M1': 12, 'N1': 24, 'K1': 35, 'M0': 46, 'N0': 70, 'K0': 46}}}

    print("Expected metrics:", metrics == corr_metrics)

    energy = run_accelergy(metrics, "simple")
    corr_energy = 193250.30592999997
    print("Expected energy:", energy == corr_energy)

def dump_counts(metrics, fn):
    local = []

    mem_counts = [{"counts": (metrics["Z"]["MainMemory"]["A"]["read"] + \
                              metrics["Z"]["MainMemory"]["B"]["read"] + \
                              metrics["Z"]["MainMemory"]["Z"]["read"]) // 8,
                   "name": "read"},
                  {"counts": metrics["Z"]["MainMemory"]["Z"]["write"] // 8,
                   "name": "write"}]
    mem = {"name": "extensor_v1_design.main_memory",
           "action_counts": mem_counts}
    local.append(mem)

    llb_counts = [{"counts": ((metrics["Z"]["LLB"]["A"]["read"] + \
                               metrics["Z"]["LLB"]["B"]["read"] + \
                               metrics["Z"]["LLB"]["Z"]["read"]) // 8 + 2048 - 1) // 2048,
                   "name": "read"},
                  {"counts": (metrics["Z"]["LLB"]["Z"]["write"] // 8 + 2048 - 1) // 2048,
                   "name": "write"}]
    llb = {"name": "extensor_v1_design.MID_SDOP.LLB",
           "action_counts": llb_counts}
    local.append(llb)

    top_counts = [{"counts": metrics["Z"]["iter"]["N2"],
                   "name": "iterate_n"},
                   {"counts": metrics["Z"]["iter"]["K2"],
                    "name": "iterate_k"},
                   {"counts": metrics["Z"]["iter"]["M2"],
                    "name": "iterate_m"}]
    top = {"name": "extensor_v1_design.TOP_SDOP.sequencer",
           "action_counts": top_counts}
    local.append(top)

    mid_counts = [{"counts": metrics["Z"]["iter"]["M1"],
                   "name": "iterate_m"},
                  {"counts": metrics["Z"]["iter"]["N1"],
                   "name": "iterate_n"},
                  {"counts": metrics["Z"]["iter"]["K1"],
                   "name": "iterate_k"},
                  {"counts": metrics["Z"]["K1Intersect"],
                   "name": "try_intersect"},
                  {"counts": metrics["Z"]["iter"]["K1"],
                   "name": "success_intersect"}]
    mid = {"name": "extensor_v1_design.MID_SDOP.midCoordinator",
           "action_counts": mid_counts}
    local.append(mid)

    mac_counts = [{"counts": metrics["Z"]["FPMul"]["mul"],
                   "name": "mac_random"}]
    mac = {"name": "extensor_v1_design.BOT_SDOP[0].mac",
           "action_counts": mac_counts}
    local.append(mac)

    bot_counts = [{"counts": metrics["Z"]["iter"]["M0"],
                   "name": "iterate_m"},
                  {"counts": metrics["Z"]["iter"]["N0"],
                   "name": "iterate_n"},
                  {"counts": metrics["Z"]["iter"]["K0"],
                   "name": "iterate_k"},
                  {"counts": metrics["Z"]["K0Intersection"],
                   "name": "try_intersect"},
                  {"counts": metrics["Z"]["iter"]["K0"],
                   "name": "success_intersect"}]
    bot = {"name": "extensor_v1_design.BOT_SDOP[0].intraCoordinator",
           "action_counts": bot_counts}
    local.append(bot)

    # trans_counts = []
    # for i in range(128):
    #     trans_counts.append(
    #         {"arguments": {"n_cols_per_row": i + 1, "n_rows": 1},
    #          "counts": 0, "name": "transfer_random"})
    #     trans_counts.append(
    #         {"arguments": {"n_cols_per_row": i + 1, "n_rows": 1},
    #          "counts": 0, "name": "transfer_repeated"})
    # trans_counts[0]["counts"] = metrics["Z"]["iter"]["K1"]
    # trans = {"name": "extensor_v1_design.MID_SDOP.mid_bot_NoC",
    #          "action_counts": trans_counts}
    # local.append(trans)

    counts = {}
    counts["action_counts"] = {}
    counts["action_counts"]["local"] = local
    counts["action_counts"]["version"] = 0.3

    dump_yaml(counts, fn)

def eval(dataset="all"):
    if dataset == "all":
        mats = ["wiki-Vote", "p2p-Gnutella31", "ca-CondMat", "poisson3Da",
                "email-Enron"]
    else:
        mats = [dataset]

    download()

    tile_sizes = {
        "wiki-Vote": (12, 64, 128, 128),
        "p2p-Gnutella31": (32, 64, 128, 128),
        "ca-CondMat": (32, 64, 96, 128),
        "poisson3Da": (32, 64, 96, 128),
        "email-Enron": (8, 256, 32, 128)}

    if not os.path.exists("../data/generated"):
        os.makedirs("../data/generated")

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    if not os.path.exists("../data/generated/extensor-energy.csv") or \
            dataset == "all":
        with open("../data/generated/extensor-energy.csv", "w") as f:
            f.write("Matrix,Energy\n")

    for mat in mats:
        A_MK = convert(mat)
        A_KM = A_MK.swizzleRanks(["K", "M"])
        B_KN = A_MK
        B_KN.setRankIds(["K", "N"])

        metrics = run(A_KM, B_KN, *tile_sizes[mat])
        energy = run_accelergy(metrics, mat)

        with open("../data/generated/extensor-energy.csv", "a") as f:
            f.write(mat + "," + str(energy) + "\n")

def run_accelergy(metrics, mat):
    counts_fn = "tmp/extensor-" + mat + "-action-counts.yaml"
    dump_counts(metrics, counts_fn)

    output = "tmp/" + mat + "-accelergy/"
    os.system("accelergy -v1  -o " + output + \
        " ../yamls/accelergy/extensor/*yaml " + \
        "../yamls/accelergy/extensor/components/*yaml " + counts_fn + \
        " > /dev/null")

    energy_dict = load_yaml(output + "energy_estimation.yaml")
    return energy_dict["energy_estimation"]["Total"]


def dump_yaml(data, fn):
    yaml = YAML(typ='safe', pure=True)
    with open(fn, "wb") as f:
        yaml.dump(data, f)

def load_yaml(fn):
    yaml = YAML(typ='safe', pure=True)
    with open(fn, 'r') as stream:
        data_loaded = yaml.load(stream)
    return data_loaded

def get_energy(dfs, src, mat):
    opt_line = dfs[src].query("Matrix == @mat")
    if len(opt_line) == 0:
        return 0

    i = opt_line.index[0]
    # Convert from pJ (10^-12 J) to mJ (10^-3 J)
    return dfs[src].at[i, "Energy"] * 10**-9


def graph_energy(pregenerated=False):
    """
    Graph of TeAAL and Reported energies
    """
    dfs = {"Reported": pd.read_csv("../data/baselines/extensor-energy.csv")}

    if pregenerated:
        dfs["TeAAL"] = pd.read_csv("../data/pregenerated/extensor-energy.csv")
    else:
        dfs["TeAAL"] = pd.read_csv("../data/generated/extensor-energy.csv")

    srcs = ["Reported", "TeAAL"]
    mats = ["wiki-Vote", "p2p-Gnutella31", "ca-CondMat", "poisson3Da",
            "email-Enron"]

    data = []
    means = [1] * len(srcs)
    for i, src in enumerate(srcs):
        for mat in mats:
            energy = get_energy(dfs, src, mat)
            means[i] += energy
            data.append([src, mat, energy])

        means[i] /= len(mats)

    for src, mean in zip(srcs, means):
        data.append([src, "AM", mean])
    mats.append("AM")

    df = pd.DataFrame(data, columns=["Source", "Matrix", "Energy (mJ)"])

    # Graph the results
    xlabels = [mat[:2] for mat in mats]
    fig = utils.graph_time(
        df,
        srcs,
        mats,
        xlabels,
        "Energy (mJ)",
        aspect_ratio=(6, 3),
        fontsize=12)

    return fig
