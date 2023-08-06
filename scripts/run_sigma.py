import os

from fibertree import Metrics, Tensor
from fibertree.model import Compute, Format, Traffic

from teaal.parse import Einsum as EinsumParser
from teaal.parse import Mapping as MappingParser
from teaal.parse import Architecture as ArchitectureParser
from teaal.parse import Bindings as BindingsParser
from teaal.parse import Format as FormatParser
from teaal.trans.hifiber import HiFiber

def run_sigma(A_KM, B_KN):
    assert A_KM.getRankIds() == ["K", "M"]
    assert B_KN.getRankIds() == ["K", "N"]

    K, M = A_KM.getShape()
    K, N = B_KN.getShape()

    fname = "../yamls/sigma.yaml"
    einsum = EinsumParser.from_file(fname)
    mapping = MappingParser.from_file(fname)
    arch = ArchitectureParser.from_file(fname)
    bindings = BindingsParser.from_file(fname)
    format_ = FormatParser.from_file(fname)

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    exec(str(HiFiber(einsum, mapping, arch, bindings, format_)), globals(), locals())

    return locals()["metrics"]

def check_sigma(metrics):
    corr = {'Z': {'DataSRAMBanks': {'A': {'read': 4096}, 'B': {'read': 28672}}, 'Multiplier': {'mul': 98}}}

    print("Expected metrics:", metrics == corr)
