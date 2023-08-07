import copy
import os

import pandas as pd

from fibertree import Metrics, Fiber, Tensor
from fibertree.model import Compute, Format, Traffic
from fibertree.model.intersect import LeaderFollowerIntersector

from teaal.parse import Einsum as EinsumParser
from teaal.parse import Mapping as MappingParser
from teaal.parse import Architecture as ArchitectureParser
from teaal.parse import Bindings as BindingsParser
from teaal.parse import Format as FormatParser
from teaal.trans.hifiber import HiFiber

from scripts.convert import convert
from scripts.download import download
import scripts.graph_utils as utils
import scripts.preprocess as preprocess

def run(A_MK, B_KN):
    assert A_MK.getRankIds() == ["M", "K"]
    assert B_KN.getRankIds() == ["K", "N"]

    M, K = A_MK.getShape()
    K, N = B_KN.getShape()
    A_MK.setFormat("M", "U")
    B_KN.setFormat("K", "U")

    fname = "../yamls/teaal/gamma.yaml"
    einsum = EinsumParser.from_file(fname)
    mapping = MappingParser.from_file(fname)
    arch = ArchitectureParser.from_file(fname)
    bindings = BindingsParser.from_file(fname)
    format_ = FormatParser.from_file(fname)

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    exec(str(HiFiber(einsum, mapping, arch, bindings, format_)), globals(), locals())

    A_MK.setFormat("M", "C")
    B_KN.setFormat("K", "C")

    return locals()["metrics"]

def check(metrics):
    corr = {'T': {'MainMemory': {'B': {'read': 2368}, 'A': {'read': 1600}}, 'Intersect': 14}, 'Z': {'MainMemory': {'Z': {'read': 0, 'write': 3456}}, 'HighRadixMerger': {'T_MKN': 60}, 'FPMul': {'mul': 46}, 'FPAdd': {'add': 13}}}

    print("Expected metrics:", metrics == corr)

def eval():
    mats = ["wiki-Vote", "p2p-Gnutella31", "ca-CondMat", "poisson3Da",
            "email-Enron"]

    download()
    preprocess.gamma()

    if not os.path.exists("../data/generated"):
        os.makedirs("../data/generated")

    with open("../data/generated/gamma.csv", "w") as f:
        f.write("Matrix,A,B,Z,Multiplies,Adds,Swaps,Minimum Traffic,Time\n")

    for mat in mats:
        A_MK = convert(mat + "-gp")
        B_KN = convert(mat)
        B_KN.setRankIds(["K", "N"])

        metrics = run(A_MK, B_KN)

        A = metrics["T"]["MainMemory"]["A"]["read"] // 8
        B = metrics["T"]["MainMemory"]["B"]["read"] // 8
        Z = metrics["Z"]["MainMemory"]["Z"]["write"] // 8

        # Bandwidth: 128 GB/s * 2^30 B/GB * 10^-9 s/ns
        mem_time = (A + B + Z) / (128 * 2**30 * 10**-9)

        # Compute Ceiling: 32 ops/cycle * 1 gigacycles/s * 10^9 cycles/gigcycle * 10^-9 s/ns
        compute_time = max(metrics["T"]["Intersect"],
            metrics["Z"]["FPMul"]["mul"],
            metrics["Z"]["FPAdd"]["add"],
            metrics["Z"]["HighRadixMerger"]["T_MKN"]) / (32 * 1 * 10**9 * 10**-9)
        time = max(mem_time, compute_time)

        df = pd.read_csv("../data/pregenerated/gamma.csv")
        i = df.query("Matrix == @mat").index[0]
        min_traffic = df.at[i, "Minimum Traffic"]

        data = [mat, A, B, Z, metrics["Z"]["FPMul"]["mul"],
                metrics["Z"]["FPAdd"]["add"], metrics["Z"]["HighRadixMerger"]["T_MKN"],
                min_traffic, time]

        with open("../data/generated/gamma.csv", "a") as f:
            f.write(",".join(str(val) for val in data) + "\n")


def norm_traffic(df, src, mat):
    i = df.query("Matrix == @mat").index[0]
    norm = df.at[i, "Minimum Traffic"]

    data = [src, mat]
    data.append(df.at[i, "A"] / norm)
    data.append(df.at[i, "B"] / norm)
    data.append(df.at[i, "Z"] / norm)

    return data

def graph_mem(pregenerated=False):
    dfs = {"Reported": pd.read_csv("../data/baselines/gamma.csv")}

    if pregenerated:
        dfs["TeAAL"] = pd.read_csv("../data/pregenerated/gamma.csv")
    else:
        dfs["TeAAL"] = pd.read_csv("../data/generated/gamma.csv")

    srcs = ["Reported", "TeAAL"]
    mats = ["wiki-Vote", "p2p-Gnutella31", "ca-CondMat", "poisson3Da",
            "email-Enron"]

    data = []
    for mat in mats:
        for src in srcs:
            data.append(norm_traffic(dfs[src], src, mat))

    df = pd.DataFrame(data, columns=["Source", "Matrix", "A", "B", "Z"])

    # Graph the results
    fig = utils.graph_traffic(df, srcs, mats, ymax=1.7)

    return fig

def get_speedup(dfs, src, mat):
    mkl_i = dfs["MKL"].query("Matrix == @mat").index[0]
    mkl_time = dfs["MKL"].at[mkl_i, "Time"]

    i = dfs[src].query("Matrix == @mat").index[0]
    time = dfs[src].at[i, "Time"]

    return mkl_time / time


def graph_speedup(pregenerated=False):
    """
    Graph of TeAAL and Reported speedups
    """

    dfs = {"Reported": pd.read_csv("../data/baselines/gamma.csv"),
        "MKL": pd.read_csv("../data/baselines/mkl.csv")}

    if pregenerated:
        dfs["TeAAL"] = pd.read_csv("../data/pregenerated/gamma.csv")
    else:
        dfs["TeAAL"] = pd.read_csv("../data/generated/gamma.csv")

    srcs = ["Reported", "TeAAL"]
    mats = ["wiki-Vote", "p2p-Gnutella31", "ca-CondMat", "poisson3Da",
            "email-Enron"]

    data = []
    for src in srcs:
        for mat in mats:
            data.append([src, mat, get_speedup(dfs, src, mat)])

    df = pd.DataFrame(data, columns=["Source", "Matrix", "Speedup"])

    # Graph the results
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
        ymax=60)

    return fig
