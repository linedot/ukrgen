# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from ..components.tile import tile
from ..models.lsc.reg import lsc_reg_type,lsc_reg_index

class reg_compare:
    def __init__(self, ttype : lsc_reg_type,
                 index : lsc_reg_index,
                 t : tile = None):

        assert isinstance(index, lsc_reg_index)
        assert isinstance(ttype, lsc_reg_type)
        self.ttype = ttype
        self.index = index
        self.t = t

    def __eq__(self, other):
        if len(self.index.indices) != len(other.index.indices):
            return False
        if self.ttype != other.ttype:
            return False
        if self.index.component != other.index.component:
            return False
        if any([idx1 != idx2 for idx1,idx2 \
                in zip(self.index.indices,other.index.indices)]):
            return False

        return True

    def __hash__(self):
        h =  hash((self.ttype,self.index))
        #print(f"hash of {self.__str__()} : {h}")
        return h
    def __str__(self):
        result = "RES:"
        if self.ttype == lsc_reg_type.address:
            result = "ADDR:"
        result += str(self.index)
        return result
    def __repr__(self):
        return self.__str__()

