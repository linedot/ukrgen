# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from typing import Self, Callable

class stridexvlen:
    """
    encodes a multiplicative chain of strides and vlens
    """
    def __init__(self, stride_ids : set[int], vlen_ids : set[int]):
        self.stride_ids = stride_ids
        self.vlen_ids = vlen_ids

    def __eq__(self, other):
        return self.stride_ids == other.stride_ids and \
               self.vlen_ids == other.vlen_ids
    def __str__(self):
        sstr = ""
        vstr = ""
        if self.stride_ids:
            sstr = "*".join([f"stride{id}" for id in self.stride_ids])
            sstr += "*"
        if self.vlen_ids:
            vstr =  "*".join([f"VLEN{id}" for id in self.vlen_ids])


        return f"{sstr}{vstr}"
    def __repr__(self):
        return str(self)
    def __hash__(self):
        return hash(str(self))

class lsc_offset:
    """Offset for addresses

    :param sxv_strides: for each complex offset, numerical multiple of that offset
    :type reg_strides: list[int]
    :param reg_strides: for each GP register containing a stride, offset in 
                        multiples of the value stored in it
    :type reg_strides: list[int]
    :param vlen_strides: for each dimension of VLEN, offset in multiples of that value,
                         for example:
                           - [a*VLEN] for 1D VLA offsets,
                           - [a*VLEN, b*RLEN] or
                             [a*(VLEN/RLEN), b*RLEN] for 2D VLA offset 
                             (exact encoding not fixed yet)
    :type vlen_strides: list[int]
    :param immoff: immediate offset in elements
    :type immoff: int

    """
    def __init__(self,
                 sxv_strides : dict[stridexvlen,int],
                 reg_strides : list[int],
                 vlen_strides : list[int],
                 immoff : int):
        self.sxv_strides = sxv_strides
        self.reg_strides = reg_strides
        self.vlen_strides = vlen_strides
        self.immoff = immoff

    @classmethod
    def adjust_offlists(cls, one, other):

        for key in one.sxv_strides.keys():
            if key not in other.sxv_strides:
                other.sxv_strides[key] = 0

        for key in other.sxv_strides.keys():
            if key not in one.sxv_strides:
                one.sxv_strides[key] = 0


        l1 = len(one.reg_strides)
        l2 = len(other.reg_strides)
        if l1 > l2:
            other.reg_strides.extend([0 for i in range(l2,l1)])
        if l2 > l1:
            one.reg_strides.extend([0 for i in range(l1,l2)])

        l1 = len(one.vlen_strides)
        l2 = len(other.vlen_strides)
        if l1 > l2:
            other.vlen_strides.extend([0 for i in range(l2,l1)])
        if l2 > l1:
            one.vlen_strides.extend([0 for i in range(l1,l2)])


    def is_scalar(self) -> bool:
        result = False
        if self.immoff != 0:
            result = True
        if any([v != 0 for v in self.vlen_strides]):
            result = False
        if any([v != 0 for k,v in self.sxv_strides.items()]):
            result = False
        if any([v != 0 for v in self.reg_strides]):
            result = False

        return result

    def is_vector(self) -> bool:
        result = False
        if 1 == sum([1 for v in self.vlen_strides if 0 != v]):
            result = True

        if any([v != 0 for k,v in self.sxv_strides.items()]):
            result = False
        if any([v != 0 for v in self.reg_strides]):
            result = False
        if self.immoff != 0:
            result = False

        return result

    def __abs__(self):
        return lsc_offset(
            sxv_strides={key : abs(val) for \
                    key,val in self.sxv_strides.items()},
            reg_strides = [abs(val) for val in self.reg_strides],
            vlen_strides = [abs(val) for val in self.vlen_strides],
            immoff = abs(self.immoff)
        )

    def __add__(self, other : Self):
        if not isinstance(other, lsc_offset):
            raise NotImplementedError(f"can't add lsc_offset and {type(other)}")

        lsc_offset.adjust_offlists(self,other)

        return lsc_offset(
            sxv_strides={key : self.sxv_strides[key]+other.sxv_strides[key]\
                    for key in self.sxv_strides.keys()},
            reg_strides=[s+o for s,o in zip(self.reg_strides,other.reg_strides)],
            vlen_strides=[s+o for s,o in zip(self.vlen_strides,other.vlen_strides)],
            immoff=self.immoff+other.immoff)


    def __sub__(self, other : Self):
        if not isinstance(other, lsc_offset):
            raise NotImplementedError(f"can't subtract {type(other)} from lsc_offset")

        lsc_offset.adjust_offlists(self,other)

        return lsc_offset(
            sxv_strides={key : self.sxv_strides[key]-other.sxv_strides[key]\
                    for key in self.sxv_strides.keys()},
            reg_strides=[s-o for s,o in zip(self.reg_strides,other.reg_strides)],
            vlen_strides=[s-o for s,o in zip(self.vlen_strides,other.vlen_strides)],
            immoff=self.immoff-other.immoff)

    def colin(self, other : Self):
        """
        Return the "colinear" part of this offset with the other offset
        All different strides/vlens/strides*vlens are treated as independent
        For example if this offset is 2*stride0*vlen1 + 3*vlen1 and
        the other offset is 1*vlen1, this returns 3*vlen1

        :param other: The other offset
        :return: Part of this offset that is colinear with other
        """
        lsc_offset.adjust_offlists(self,other)

        result = lsc_offset.zero_offset()

        for key in self.sxv_strides:
            if (0 != self.sxv_strides[key]) and (0 != other.sxv_strides[key]):
                result.sxv_strides[key] = self.sxv_strides[key]

        result.reg_strides = [s if (s!=0 and o!=0) else 0 for s,o in \
                zip(self.reg_strides,other.reg_strides)]

        result.vlen_strides = [s if (s!=0 and o!=0) else 0 for s,o in \
                zip(self.vlen_strides,other.vlen_strides)]

        if (self.immoff != 0) and (other.immoff != 0):
            result.immoff = self.immoff

        return result

    def allcompare(self, other : Self,
                   comparison : Callable[[Self,Self],bool]):
        
        lsc_offset.adjust_offlists(self,other)

        return all([comparison(self.sxv_strides[key],other.sxv_strides[key]) \
                for key in self.sxv_strides]) and \
               all([comparison(s,o) for s,o in \
                   zip(self.reg_strides,other.reg_strides)]) and \
               all([comparison(s,o) for s,o in \
                   zip(self.vlen_strides, other.vlen_strides)]) and \
               (comparison(self.immoff,other.immoff))

    def anycompare(self, other : Self,
                   comparison : Callable[[Self,Self],bool]):
        
        lsc_offset.adjust_offlists(self,other)

        return any([comparison(self.sxv_strides[key],other.sxv_strides[key]) \
                for key in self.sxv_strides]) or \
               any([comparison(s,o) for s,o in \
                   zip(self.reg_strides,other.reg_strides)]) or \
               any([comparison(s,o) for s,o in \
                   zip(self.vlen_strides, other.vlen_strides)]) or \
               (comparison(self.immoff,other.immoff))


    def __lt__(self, other : Self):
        # NOTE: this is sketchy. It's used in addr_resolver,
        #       perhaps this should be removed and a more elaborate check that
        #       takes the required offset to target into account should be 
        #       implemented?
        #return self.allcompare(other, lambda s,o : s <= o) and \
        #       self.anycompare(other, lambda s,o : s < o)
        lsc_offset.adjust_offlists(self,other)


        # NOTE: For this the assumptions are:
        #       s > sxv > v > i
        #       This somewhat makes sense in the context of GEMM kernels,
        #       but not the general case

        if all([s <= o for s,o in zip(self.reg_strides,other.reg_strides)]) and \
           any([s < o for s,o in zip(self.reg_strides,other.reg_strides)]):
            return True
        if any([s > o for s,o in zip(self.reg_strides,other.reg_strides)]):
            return False

        if all([self.sxv_strides[key] <= other.sxv_strides[key] \
                for key in self.sxv_strides]) and \
           any([self.sxv_strides[key] < other.sxv_strides[key] \
                for key in self.sxv_strides]):
            return True
        if any([self.sxv_strides[key] > other.sxv_strides[key] \
                for key in self.sxv_strides]):
            return False

        if all([s <= o for s,o in zip(self.vlen_strides,other.vlen_strides)]) and \
           any([s < o for s,o in zip(self.vlen_strides,other.vlen_strides)]):
            return True
        if any([s > o for s,o in zip(self.vlen_strides,other.vlen_strides)]):
            return False


        return self.immoff < other.immoff


    def __eq__(self, other : Self):
        return self.allcompare(other, lambda s,o : s == o)

    def __str__(self):
        result = ""
        if self.sxv_strides:
            rstr = "+".join([f"{val}*{key}" for key,val \
                    in self.sxv_strides.items() if val != 0])
            if rstr:
                result += f"({rstr})+"
        if self.reg_strides:
            rstr = "+".join([f"{o}*stride{i}" for i,o in enumerate(self.reg_strides) if o != 0])
            if rstr:
                result += f"({rstr})+"
        if self.vlen_strides:
            vstr = "+".join([f"{o}*VLEN{i+1}" for i,o in enumerate(self.vlen_strides) if o != 0])
            if vstr:
                result += f"({vstr})+"
        
        return f"{result}{self.immoff}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.__str__())

    @classmethod
    def zero_offset(cls):
        return cls(dict(),[],[],0)

    @classmethod
    def vo(cls, vid : int, value : int):
        """
        1D Vector offset
        """
        voffs = [0 for i in range(vid+1)]
        voffs[vid] = value
        return cls(dict(),[],voffs,0)

    @classmethod
    def to(cls, vid1 : int, vid2 : int, value1 : int, value2):
        """
        2D Tile offset
        """
        voffs = [0 for i in range(max(vid1,vid2)+1)]
        voffs[vid1] = value1
        voffs[vid2] = value2
        return cls(dict(),[],voffs,0)

    @classmethod
    def so(cls, value : int):
        """
        scalar offset
        """
        return cls(dict(),[],[],value)
