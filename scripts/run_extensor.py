import os

from fibertree import Metrics, Tensor
from fibertree.model import Compute, Format, Traffic
from fibertree.model.intersect import SkipAheadIntersector

from teaal.parse import Einsum as EinsumParser
from teaal.parse import Mapping as MappingParser
from teaal.parse import Architecture as ArchitectureParser
from teaal.parse import Bindings as BindingsParser
from teaal.parse import Format as FormatParser
from teaal.trans.hifiber import HiFiber

def run_extensor(A_KM, B_KN, M1, K1, N1, pe_sz):
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

    fname = "../yamls/extensor.yaml"
    einsum = EinsumParser.from_file(fname)
    mapping = MappingParser.from_file(fname)
    arch = ArchitectureParser.from_file(fname)
    bindings = BindingsParser.from_file(fname)
    format_ = FormatParser.from_file(fname)

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    exec(str(HiFiber(einsum, mapping, arch, bindings, format_)), globals(), locals())

    return locals()["metrics"]

def check_extensor(metrics):
    corr = {'Z': {'MainMemory': {'A': {'read': 7296}, 'B': {'read': 6464}, 'Z': {'read': 2560, 'write': 6912}}, 'FPMul': {'mul': 46}, 'FPAdd': {'add': 13}, 'K2Intersect': 4, 'K1Intersect': 39, 'K0Intersection': 84}}

    print("Expected metrics:", metrics == corr)
