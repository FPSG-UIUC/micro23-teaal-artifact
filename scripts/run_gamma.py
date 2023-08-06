import os

from fibertree import Metrics, Fiber, Tensor
from fibertree.model import Compute, Format, Traffic
from fibertree.model.intersect import LeaderFollowerIntersector

from teaal.parse import Einsum as EinsumParser
from teaal.parse import Mapping as MappingParser
from teaal.parse import Architecture as ArchitectureParser
from teaal.parse import Bindings as BindingsParser
from teaal.parse import Format as FormatParser
from teaal.trans.hifiber import HiFiber

def run_gamma(A_MK, B_KN):
    assert A_MK.getRankIds() == ["M", "K"]
    assert B_KN.getRankIds() == ["K", "N"]

    M, K = A_MK.getShape()
    K, N = B_KN.getShape()
    A_MK.setFormat("M", "U")
    B_KN.setFormat("K", "U")

    fname = "../yamls/gamma.yaml"
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

def check_gamma(metrics):
    corr = {'T': {'MainMemory': {'B': {'read': 2368}, 'A': {'read': 1600}}, 'Intersect': 14}, 'Z': {'MainMemory': {'Z': {'read': 0, 'write': 3456}}, 'HighRadixMerger': {'T_MKN': 60}, 'FPMul': {'mul': 46}, 'FPAdd': {'add': 13}}}

    print("Expected metrics:", metrics == corr)
