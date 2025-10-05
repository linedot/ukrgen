# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

class lsc_reg_index:
    """
    index for an lsc register

    multiple integers are for encoding something like lanes within the register,
    for example if an ARM NEON register v3 contained 4 FP32 values, the second
    lane, i.e. v3.s[1] would mean indices=[3,1]. More than 2 elements in indices
    could be useful for an ISA with ND registers with addressable elements
    """
    def __init__(self, component : str, indices : list[int]):
        if not isinstance(component, str):
            raise ValueError(f"Component {component} is not str")
        for i in indices:
            if not isinstance(i, int):
                raise ValueError(f"Non-numeric element {i} in indices list")
        self.component = component
        self.indices = indices
    

    def __eq__(self, other):
        return self.component == other.component and \
                  all([i1 == i2 for i1,i2 in zip(
                      self.indices, other.indices
                      )])

    def __str__(self) -> str:
        """
        Converts an LSC register index to a string
        """
        iicount = len(self.indices)
        if iicount > 2 or iicount < 1:
            # >2 hasn't been fleshed out yet, so error out for now
            raise ValueError(f"Invalid number of numeric indices in index: {iicount}")
        if iicount == 2:
            return f"{self.component}{self.indices[0]}.el{self.indices[1]}"
        if iicount == 1:
            return f"{self.component}{self.indices[0]}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash(str(self))
