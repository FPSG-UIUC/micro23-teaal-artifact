import os
import random

import pandas as pd

from fibertree import Metrics, Tensor
from fibertree.model import Compute, Format, Traffic

from teaal.parse import Einsum as EinsumParser
from teaal.parse import Mapping as MappingParser
from teaal.parse import Architecture as ArchitectureParser
from teaal.parse import Bindings as BindingsParser
from teaal.parse import Format as FormatParser
from teaal.trans.hifiber import HiFiber

from scripts.convert import convert
from scripts.download import download
import scripts.graph_utils as utils

def run(A_KM, B_KN):
    assert A_KM.getRankIds() == ["K", "M"]
    assert B_KN.getRankIds() == ["K", "N"]

    K, M = A_KM.getShape()
    K, N = B_KN.getShape()
    A_KM.setFormat("K", "U")
    B_KN.setFormat("K", "U")

    fname = "../yamls/teaal/outerspace.yaml"
    einsum = EinsumParser.from_file(fname)
    mapping = MappingParser.from_file(fname)
    arch = ArchitectureParser.from_file(fname)
    bindings = BindingsParser.from_file(fname)
    format_ = FormatParser.from_file(fname)

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    exec(str(HiFiber(einsum, mapping, arch, bindings, format_)), globals(), locals())

    A_KM.setFormat("K", "C")
    B_KN.setFormat("K", "C")

    return locals()["metrics"]

def check(metrics):
    corr = {'T0': {'MainMemory': {'B': {'read': 2368}, 'A': {'read': 1664}}, 'FPMul': {'mul': 46}}, 'T1': {'MainMemory': {'T0': {'read': 5184}, 'T1': {'read': 0, 'write': 5184}}}, 'Z': {'MainMemory': {'Z': {'read': 0, 'write': 3456}}, 'SortHW': {'T1_MKN': 71}, 'FPAdd': {'add': 13}}}

    print("Expected metrics:", metrics == corr)

def prepare_output_file(mats):
    if not os.path.exists("../data/generated"):
        os.makedirs("../data/generated")

    # Create the file if one does not exist
    if not os.path.exists("../data/generated/outerspace.csv"):
        with open("../data/generated/outerspace.csv", "w") as f:
            f.write("Matrix,A,B,T,Z,Multiplies,Adds,Swaps,Minimum Traffic,Time,Seed\n")

    # Otherwise remove matching rows
    else:
        with open("../data/generated/outerspace.csv", "r") as old, \
                 open("../data/generated/outerspace-tmp.csv", "w") as new:
            line = old.readline()
            while line:
                split = line.split(",")
                if split[0] not in mats:
                    new.write(line)

                line = old.readline()

        os.replace("../data/generated/outerspace-tmp.csv", "../data/generated/outerspace.csv")

def write_data(mat, metrics, seed=0):
    A = metrics["T0"]["MainMemory"]["A"]["read"] // 8
    B = metrics["T0"]["MainMemory"]["B"]["read"] // 8
    T = metrics["T1"]["MainMemory"]["T0"]["read"] // 8 + \
        metrics["T1"]["MainMemory"]["T1"]["write"] // 8
    Z = metrics["Z"]["MainMemory"]["Z"]["write"]

    # Bandwidth: 128 GB/s * 2^30 B/GB * 10^-9 ns/s
    mem_time = (A + B + T + Z) / (128 * 2**30 * 10**-9)

    # Compute Ceiling: # PEs ops/cycle * 1.5 gigacycles/s * 10^9 cycles/gigcycle * 10^-9 s/ns
    compute_time = metrics["T0"]["FPMul"]["mul"] / (256 * 1.5 * 10**9 * 10**-9) + \
        max(metrics["Z"]["FPAdd"]["add"],
            metrics["Z"]["SortHW"]["T1_MKN"]) / (128 * 1.5 * 10**9 * 10**-9)
    time = max(mem_time, compute_time)

    df = pd.read_csv("../data/pregenerated/outerspace.csv")
    i = df.query("Matrix == @mat").index[0]
    min_traffic = df.at[i, "Minimum Traffic"]

    data = [mat, A, B, T, Z, metrics["T0"]["FPMul"]["mul"],
            metrics["Z"]["FPAdd"]["add"], metrics["Z"]["SortHW"]["T1_MKN"],
            min_traffic, time, seed]

    with open("../data/generated/outerspace.csv", "a") as f:
        f.write(",".join(str(val) for val in data) + "\n")



def eval_mem():
    mats = ["wiki-Vote", "p2p-Gnutella31", "ca-CondMat", "poisson3Da",
            "email-Enron"]

    download()
    prepare_output_file(mats)

    for mat in mats:
        A_MK = convert(mat)
        A_KM = A_MK.swizzleRanks(["K", "M"])
        B_KN = A_MK
        B_KN.setRankIds(["K", "N"])

        metrics = run(A_KM, B_KN)

        write_data(mat, metrics)

def eval_time():
    mats = ["uniform" + str(i) for i in range(5)]
    prepare_output_file(mats)

    mat_configs = {
        "uniform0": (4986, 8 * 10**-3),
        "uniform1": (9987, 2 * 10**-3),
        "uniform2": (19937, 5 * 10**-4),
        "uniform3": (39888, 1.3 * 10**-4),
        "uniform4": (79730, 3.1 * 10**-5)
    }

    for mat in mats:
        dim, density = mat_configs[mat]

        seed = random.random()

        A_KM = Tensor.fromRandom(
            rank_ids=["K", "M"], shape=[dim, dim],
            density=[1.0, density], seed=seed)

        B_KN = A_KM.swizzleRanks(["M", "K"])
        B_KN.setRankIds(["K", "N"])

        metrics = run(A_KM, B_KN)
        write_data(mat, metrics, seed)

def norm_traffic(df, src, mat):
    i = df.query("Matrix == @mat").index[0]
    norm = df.at[i, "Minimum Traffic"]

    data = [src, mat]
    data.append(df.at[i, "A"] / norm)
    data.append(df.at[i, "B"] / norm)
    data.append(df.at[i, "Z"] / norm)
    data.append(df.at[i, "T"] / norm)

    return data

def graph_mem(pregenerated=False):
    dfs = {"Reported": pd.read_csv("../data/baselines/outerspace.csv")}

    if pregenerated:
        dfs["TeAAL"] = pd.read_csv("../data/pregenerated/outerspace.csv")
    else:
        dfs["TeAAL"] = pd.read_csv("../data/generated/outerspace.csv")

    srcs = ["Reported", "TeAAL"]
    mats = ["wiki-Vote", "p2p-Gnutella31", "ca-CondMat", "poisson3Da",
            "email-Enron"]

    data = []
    for mat in mats:
        for src in srcs:
            data.append(norm_traffic(dfs[src], src, mat))

    df = pd.DataFrame(data, columns=["Source", "Matrix", "A", "B", "Z", "T"])

    fig = utils.graph_traffic(df, srcs, mats)

    return fig

def get_time(dfs, src, mat):
    i = dfs[src].query("Matrix == @mat").index[0]
    return dfs[src].at[i, "Time"] * 10**-9

def graph_time(pregenerated=False):
    dfs = {"Reported": pd.read_csv("../data/baselines/outerspace.csv")}

    if pregenerated:
        dfs["TeAAL"] = pd.read_csv("../data/pregenerated/outerspace.csv")
    else:
        dfs["TeAAL"] = pd.read_csv("../data/generated/outerspace.csv")

    srcs = ["Reported", "TeAAL"]
    mats = ["uniform" + str(i) for i in range(5)]

    xlabels = ["4,986/8.0E-3", "9,987/2.0E-3", "19,937/5.0E-4",
               "39,888/1.3E-4", "79,730/3.1E-5"]
    xrotation = 30
    xlabel = "Dimension/Density"

    data = []
    for src in srcs:
        for mat in mats:
            data.append([src, mat, get_time(dfs, src, mat)])

    df = pd.DataFrame(data, columns=["Source", "Matrix", "Time"])

    fig = utils.graph_time(
        df,
        srcs,
        mats,
        xlabels,
        "Execution Time (s)",
        xlabel=xlabel,
        xrotation=xrotation)

    return fig
