from typing import Callable
from abc import abstractmethod

from .load_store_operations import lsc_offset
from ..components import tile,dimension_type

class tile_offset_mapper:
    @abstractmethod
    def __call__(
            tile : tile,
            idx : tuple[int,int]) -> lsc_offset:
# I don't think we need the subindex for data origins
#            subidx : int) -> int:
        raise NotImplementedError("tried calling abstract tile offset mapper")


class flat_mapper(tile_offset_mapper):

    def __init__(self, get_flat_idx : Callable[[tile,tuple[int,int]],int]):
        self.get_flat_idx=get_flat_idx

    def get_idx(self, t : tile, flat_idx : int):
        if (t.dima.dt == dimension_type.vla) != \
           (t.dimb.dt == dimension_type.vla):
               return lsc_offset({},[],[flat_idx],0)
        elif (t.dima.dt == dimension_type.vla) and \
             (t.dimb.dt == dimension_type.vla):
               return lsc_offset({},[],[0,flat_idx],0)
        elif (t.dima.dt != dimension_type.vla) and \
             (t.dimb.dt != dimension_type.vla):
               return lsc_offset({},[],[],flat_idx)

        raise ValueError("Unhandled case in tile offset mapper")

    def __call__(self, t : tile,
                 idx : tuple[int,int]) -> lsc_offset:
        flat_idx = self.get_flat_idx(t,idx)
        return self.get_idx(t,flat_idx)
