import os

from fibertree import Metrics, Tensor
from fibertree.model import Compute, Format, Traffic

from teaal.parse import Einsum as EinsumParser
from teaal.parse import Mapping as MappingParser
from teaal.parse import Architecture as ArchitectureParser
from teaal.parse import Bindings as BindingsParser
from teaal.parse import Format as FormatParser
from teaal.trans.hifiber import HiFiber

def run_outerspace(A_KM, B_KN):
    assert A_KM.getRankIds() == ["K", "M"]
    assert B_KN.getRankIds() == ["K", "N"]

    K, M = A_KM.getShape()
    K, N = B_KN.getShape()
    A_KM.setFormat("K", "U")
    B_KN.setFormat("K", "U")

    fname = "../yamls/outerspace.yaml"
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

def check_outerspace(metrics):
    corr = {'T0': {'MainMemory': {'B': {'read': 2368}, 'A': {'read': 1664}}, 'FPMul': {'mul': 46}}, 'T1': {'MainMemory': {'T0': {'read': 5184}, 'T1': {'read': 0, 'write': 5184}}}, 'Z': {'MainMemory': {'Z': {'read': 0, 'write': 3456}}, 'SortHW': {'T1_MKN': 71}, 'FPAdd': {'add': 13}}}

    print("Expected metrics:", metrics == corr)
