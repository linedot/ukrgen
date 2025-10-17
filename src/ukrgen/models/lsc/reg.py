from enum import Enum,auto

from .index import lsc_reg_index

class lsc_reg_type(Enum):
    address = auto()
    data = auto()
    offset = auto()
    value = auto()

class lsc_reg(lsc_reg_index):

    """
    lsc register

    combination of index and register type
    """
    def __init__(self, component : str,
                 indices: list[int],
                 rtype : lsc_reg_type):
        self.rtype = rtype
        super().__init__(component=component, indices=indices)

    def __eq__(self, other):
        return (self.rtype == other.rtype) and super().__eq__(other)

    def __str__(self):
        tstr = ""
        if self.rtype == lsc_reg_type.address:
            tstr = "ADDR:"
        elif self.rtype == lsc_reg_type.data:
            tstr = "RES:"
        elif self.rtype == lsc_reg_type.offset:
            tstr = "OFF:"
        elif self.rtype == lsc_reg_type.value:
            tstr = "VAL:"
        else:
            raise ValueError(f"Invalid rtype: {self.rtype}")
        return tstr+super().__str__() 

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash(str(self))

    @property
    def index(self) -> lsc_reg_index:
        """Get the index without register type"""
        return lsc_reg_index(self.component, self.indices)
   
