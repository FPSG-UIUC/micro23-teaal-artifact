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

    # print(HiFiber(einsum, mapping, arch, bindings, format_))
    exec(str(HiFiber(einsum, mapping, arch, bindings, format_)), globals(), locals())
    
    return locals()["metrics"]