import os

import pandas as pd

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

    fname = "../yamls/teaal/extensor.yaml"
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
    corr = {'Z': {'MainMemory': {'A': {'read': 7296}, 'B': {'read': 6464}, 'Z': {'read': 2560, 'write': 6912}}, 'FPMul': {'mul': 46}, 'FPAdd': {'add': 13}, 'K2Intersect': 4, 'K1Intersect': 39, 'K0Intersect': 84}}

    print("Expected metrics:", metrics == corr)

def eval():
    mats = ["wiki-Vote", "p2p-Gnutella31", "ca-CondMat", "poisson3Da",
            "email-Enron"]

    download()

    tile_sizes = {
        "wiki-Vote": (12, 64, 128, 128),
        "p2p-Gnutella31": (32, 64, 128, 128),
        "ca-CondMat": (32, 64, 96, 128),
        "poisson3Da": (32, 64, 96, 128),
        "email-Enron": (8, 256, 32, 128)}

    if not os.path.exists("../data/generated"):
        os.makedirs("../data/generated")

    with open("../data/generated/extensor.csv", "w") as f:
        f.write("Matrix,A,B,PO,Z,Multiplies,Adds,Intersects,Minimum Traffic,Time\n")

    for mat in mats:
        A_MK = convert(mat)
        A_KM = A_MK.swizzleRanks(["K", "M"])
        B_KN = A_MK
        B_KN.setRankIds(["K", "N"])

        metrics = run(A_KM, B_KN, *tile_sizes[mat])

        A = metrics["Z"]["MainMemory"]["A"]["read"] // 8
        B = metrics["Z"]["MainMemory"]["B"]["read"] // 8
        Z_read = metrics["Z"]["MainMemory"]["Z"]["read"] // 8
        Z_write = metrics["Z"]["MainMemory"]["Z"]["write"] // 8
        Z = Z_write - Z_read
        PO = Z_read * 2

        # Bandwidth: 68.256 GB/s * 2^30 B/GB * 10^-9 s/ns
        mem_time = (A + B + PO + Z) / (68.256 * 2**30) * 10**9

        # Compute Ceiling: 128 ops/cycle * 1 gigacycles/s * 10^9 cycles/gigcycle * 10^-9 s/ns
        compute_time = max(metrics["Z"]["FPMul"]["mul"],
            metrics["Z"]["FPAdd"]["add"],
            metrics["Z"]["K2Intersect"],
            metrics["Z"]["K1Intersect"],
            metrics["Z"]["K0Intersect"]) / (128 * 10**9 * 10**-9)
        time = max(mem_time, compute_time)

        df = pd.read_csv("../data/pregenerated/extensor.csv")
        i = df.query("Matrix == @mat").index[0]
        min_traffic = df.at[i, "Minimum Traffic"]

        data = [mat, A, B, PO, Z, metrics["Z"]["FPMul"]["mul"],
                metrics["Z"]["FPAdd"]["add"], metrics["Z"]["K0Intersect"],
                min_traffic, time]

        with open("../data/generated/extensor.csv", "a") as f:
            f.write(",".join(str(val) for val in data) + "\n")


def run_sparseloop():
    mats = ["wiki-Vote", "p2p-Gnutella31", "ca-CondMat", "poisson3Da",
            "email-Enron"]

    if not os.path.exists("../data/generated"):
        os.makedirs("../data/generated")

    with open("../data/generated/sparseloop.csv", "w") as f:
        f.write("Matrix,Time\n")

    for mat in mats:
        # The matrix dimensions of p2p-Gnutella31 are too large, causing
        # Sparseloop an integer overflow error
        if mat == "p2p-Gnutella31":
            with open("../data/generated/sparseloop.csv", "a") as f:
                f.write(mat + ",0\n")
            continue

        if not os.path.exists("tmp/" + mat):
            os.makedirs("tmp/" + mat)

        yamls = "../yamls/sparseloop/extensor/" + mat + "/problem.yaml " + \
            "../yamls/sparseloop/extensor/" + mat + "/mapping.yaml " + \
            "../yamls/sparseloop/extensor/" + mat + "/sparse-opt.yaml "

        # Because Sparseloop first generates the dense traffic, the email-Enron
        # tile does not fit in the correctly-sized LLB
        if mat == "email-Enron":
            yamls += "../yamls/sparseloop/extensor/email-Enron/arch.yaml "
        else:
            yamls += "../yamls/sparseloop/extensor/arch.yaml "


        os.system("timeloop-model " + yamls + "-o tmp/" + mat)

        with open("tmp/" + mat + "/timeloop-model.stats.txt", "r") as f:
            line = f.readline()
            while line[:8] != "Cycles: ":
                line = f.readline()

            # ExTensor clock frequency is 1GHz, so cycles == nanoseconds
            cycles = line[8:-1]

        with open("../data/generated/sparseloop.csv", "a") as f:
            f.write(mat + "," + cycles + "\n")

def norm_traffic(df, src, mat):
    i = df.query("Matrix == @mat").index[0]
    norm = df.at[i, "Minimum Traffic"]

    data = [src, mat]
    data.append(df.at[i, "A"] / norm)
    data.append(df.at[i, "B"] / norm)
    data.append(df.at[i, "Z"] / norm)
    data.append(df.at[i, "PO"] / norm)

    return data

def graph_mem(pregenerated=False):
    dfs = {"Reported": pd.read_csv("../data/baselines/extensor.csv")}

    if pregenerated:
        dfs["TeAAL"] = pd.read_csv("../data/pregenerated/extensor.csv")
    else:
        dfs["TeAAL"] = pd.read_csv("../data/generated/extensor.csv")

    srcs = ["Reported", "TeAAL"]
    mats = ["wiki-Vote", "p2p-Gnutella31", "ca-CondMat", "poisson3Da",
            "email-Enron"]

    data = []
    for mat in mats:
        for src in srcs:
            data.append(norm_traffic(dfs[src], src, mat))

    df = pd.DataFrame(data, columns=["Source", "Matrix", "A", "B", "Z", "PO"])

    # Graph the results
    fig = utils.graph_traffic(df, srcs, mats)

    return fig

def get_speedup(dfs, src, mat):
    mkl_i = dfs["MKL"].query("Matrix == @mat").index[0]
    mkl_time = dfs["MKL"].at[mkl_i, "Time"]

    i = dfs[src].query("Matrix == @mat").index[0]
    time = dfs[src].at[i, "Time"]

    if time == 0:
        return 0

    else:
        return mkl_time / time

def graph_speedup(pregenerated=False):
    dfs = {"Reported": pd.read_csv("../data/baselines/extensor.csv"),
        "MKL": pd.read_csv("../data/baselines/mkl.csv")}

    if pregenerated:
        dfs["TeAAL"] = pd.read_csv("../data/pregenerated/extensor.csv")
        dfs["Sparseloop"] = pd.read_csv("../data/pregenerated/sparseloop.csv")
    else:
        dfs["TeAAL"] = pd.read_csv("../data/generated/extensor.csv")
        dfs["Sparseloop"] = pd.read_csv("../data/generated/sparseloop.csv")

    srcs = ["Reported", "TeAAL", "Sparseloop"]
    mats = ["wiki-Vote", "p2p-Gnutella31", "ca-CondMat", "poisson3Da",
            "email-Enron"]

    data = []
    for src in srcs:
        for mat in mats:
            data.append([src, mat, get_speedup(dfs, src, mat)])

    df = pd.DataFrame(data, columns=["Source", "Matrix", "Speedup"])
    width = 0.3

    xlabels = [mat[:2] for mat in mats]
    fig = utils.graph_time(
        df,
        srcs,
        mats,
        xlabels,
        "Speedup Over MKL",
        aspect_ratio=(
            3,
            3.6),
        fontsize=12,
        width=width)

    return fig
