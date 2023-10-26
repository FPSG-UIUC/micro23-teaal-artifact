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

import scripts.graph_utils as utils

def run(A_KM, B_KN):
    assert A_KM.getRankIds() == ["K", "M"]
    assert B_KN.getRankIds() == ["K", "N"]

    K, M = A_KM.getShape()
    K, N = B_KN.getShape()

    fname = "../yamls/teaal/sigma.yaml"
    einsum = EinsumParser.from_file(fname)
    mapping = MappingParser.from_file(fname)
    arch = ArchitectureParser.from_file(fname)
    bindings = BindingsParser.from_file(fname)
    format_ = FormatParser.from_file(fname)

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    # Remove the inner loop to improve performance
    hifiber = str(HiFiber(einsum, mapping, arch, bindings, format_))
    hifiber_lines = hifiber.split("\n")
    hifiber_lines.remove("            for (m, k0), a_val in a_mk00:")
    hifiber_lines.remove("                z_ref = z_m.getPayloadRef(m, trace=\"get_payload_Z\")")
    hifiber_lines.remove("                b_val = b_k0.getPayload(k0, trace=\"get_payload_B\")")
    hifiber_lines.remove("                z_ref += a_val * b_val")
    hifiber = "\n".join(hifiber_lines)

    exec(hifiber, globals(), locals())

    return locals()["metrics"]

def check(metrics):
    corr = {'Z': {'DataSRAMBanks': {'A': {'read': 4096}, 'B': {'read': 28672}, 'time': 3.973642985026042e-09}}, 'blocks': [['Z']], 'time': 3.973642985026042e-09}

    print("Expected metrics:", metrics == corr)

def eval():
    mats = [(128, 2048, 4096), (320, 3072, 4096), (1632, 36548, 1024), (2048, 4096, 32),
            (35, 8457, 2560), (31999, 1024, 84), (84, 1024, 4096), (2048, 1, 128), (256, 256, 2048)]

    if not os.path.exists("../data/generated"):
        os.makedirs("../data/generated")

    with open("../data/generated/sigma.csv", "w") as f:
        f.write("M,N,K,A,B,Time,Seed\n")

    for mat in mats:
        M, N, K = mat

        A_density = 0.2
        B_density = 0.9

        seed = random.random()

        A_KM = Tensor.fromRandom(
            rank_ids=["K", "M"], shape=[K, M],
            density=[1.0, A_density], seed=seed)

        B_KN = Tensor.fromRandom(
            rank_ids=["K", "N"], shape=[K, N],
            density=[1.0, B_density], seed=seed * 2)

        metrics = run(A_KM, B_KN)

        A = metrics["Z"]["DataSRAMBanks"]["A"]["read"] // 8
        B = metrics["Z"]["DataSRAMBanks"]["B"]["read"] // 8

        time = metrics["time"]

        data = [*mat, A, B, time, seed]

        with open("../data/generated/sigma.csv", "a") as f:
            f.write(",".join(str(val) for val in data) + "\n")



def get_speedup(dfs, src, mat):
    M, N, K = mat
    tpu_i = dfs["TPU"].query("M == @M and N == @N and K == @K").index[0]
    tpu_time = dfs["TPU"].at[tpu_i, "Time"]

    i = dfs[src].query("M == @M and N == @N and K == @K").index[0]
    time = dfs[src].at[i, "Time"]

    return tpu_time / time


def graph_speedup(pregenerated=False):
    """
    Graph of TeAAL and Reported speedups
    """

    dfs = {"Reported": pd.read_csv("../data/baselines/sigma.csv"),
        "TPU": pd.read_csv("../data/baselines/tpu.csv")}

    if pregenerated:
        dfs["TeAAL"] = pd.read_csv("../data/pregenerated/sigma.csv")
    else:
        dfs["TeAAL"] = pd.read_csv("../data/generated/sigma.csv")

    srcs = ["Reported", "TeAAL"]
    mats = [(128, 2048, 4096), (320, 3072, 4096), (1632, 36548, 1024), (2048, 4096, 32),
            (35, 8457, 2560), (31999, 1024, 84), (84, 1024, 4096), (2048, 1, 128), (256, 256, 2048)]


    data = []
    for src in srcs:
        for mat in mats:
            data.append([src, mat, get_speedup(dfs, src, mat)])

    df = pd.DataFrame(data, columns=["Source", "Matrix", "Speedup"])

    # Graph the results
    xlabels = ["/".join([str(i) for i in mat]) for mat in mats]
    xlabel = "Workload Dimensions M/N/K"
    fig = utils.graph_time(
        df,
        srcs,
        mats,
        xlabels,
        "Speedup Over TPU",
        xlabel=xlabel,
        xtick_sz=9,
        xrotation=30)

    return fig
