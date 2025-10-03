from typing import Callable
from abc import abstractmethod

from .load_store_operations import lsc_offset,stridexvlen
from ..components import tile,dimension_type


class offset_mapper:

    NIE_MESSAGE = "Reached abstract offset mapper method"

    @abstractmethod
    def get_ldst_size(self, t : tile):
        raise NotImplementedError(self.NIE_MESSAGE)

    @abstractmethod
    def map_tile_idx(
        t : tile,
        idx : tuple[int,int]) -> lsc_offset:
        # I don't think we need the subindex for data origins
        #            subidx : int) -> int:
        raise NotImplementedError(self.NIE_MESSAGE)


class flat_mapper(offset_mapper):

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

    def get_ldst_size(self, t : tile):
        return self.get_idx(t, t.dima.size*t.dimb.size)


    def map_tile_idx(self, t : tile,
                     idx : tuple[int,int]) -> lsc_offset:
        flat_idx = self.get_flat_idx(t,idx)
        return self.get_idx(t,flat_idx)


class strided_mapper(offset_mapper):

    def __init__(self,
                 dims : tuple[int,int],
                 stride_indices : tuple[None|int,
                                        None|int],
                 flip_tile_dims : bool = False,
                 vecdim : int = 0):
        self.dims = dims
        self.stride_indices = stride_indices
        self.flip_tile_dims = flip_tile_dims
        self.vecdim = vecdim


    def get_ldst_size(self, t : tile) -> lsc_offset:
        # TODO: When we start implementing arbitrary vectorization directions,
        #       This will probably need a direction parameter
        #       Alternatively this could be handled in some other manner so that
        #       always using the first component in the mapper is correct
        return self.map_tile_idx(t=t, idx=(1,0))

    def map_tile_idx(self, t : tile, idx : tuple[int,int]) -> lsc_offset:

        first_t_size = t.dima.size
        second_t_size = t.dimb.size
        if self.flip_tile_dims:
            first_t_size = t.dimb.size
            second_t_size = t.dima.size

        first  = first_t_size*idx[0]
        second = second_t_size*idx[1]
        
        result = lsc_offset.zero_offset()

        s0 = self.stride_indices[0]
        s1 = self.stride_indices[1]

        istile = (t.dima.dt == dimension_type.vla) and \
                 (t.dimb.dt == dimension_type.vla)

        isvector = (t.dima.dt == dimension_type.vla) != \
                   (t.dimb.dt == dimension_type.vla)

        isscalar = (t.dima.dt != dimension_type.vla) and \
                   (t.dimb.dt != dimension_type.vla)


        #if istile:
        #    raise NotImplementedError("Tiles not implemented in mapper")


        first_off_base = lsc_offset.zero_offset()
        second_off_base = lsc_offset.zero_offset()

        # first dim
        if s0 is not None:
            if t.dima.dt == dimension_type.vla:
                first_off_base = lsc_offset(
                            {
                                stridexvlen({s0},{0}) : 1
                            },
                            [],[],0
                        )
            else:
                stridelist = [0 for i in range(s0+1)]
                stridelist[s0] = 1
                first_off_base = lsc_offset({},stridelist,[],0)
        else:
            if t.dima.dt == dimension_type.vla:
                first_off_base = lsc_offset({}, [], [1], 0)
            else:
                first_off_base = lsc_offset({}, [], [], 1)

        # second dim
        if s1 is not None:
            if t.dimb.dt == dimension_type.vla:
                second_off_base = lsc_offset(
                            {
                                stridexvlen({s1},{0}) : 1
                            },
                            [],[],0
                        )
            else:
                stridelist = [0 for i in range(s1+1)]
                stridelist[s1] = 1
                second_off_base = lsc_offset({},stridelist,[],0)
        else:
            if t.dima.dt == dimension_type.vla:
                second_off_base = lsc_offset({}, [], [1], 0)
            else:
                second_off_base = lsc_offset({}, [], [], 1)

        if s1 is None and 0 == self.vecdim:
            first_size = self.dims[0]*first_t_size
            second_off_base = sum([first_off_base for i in range(first_size)],lsc_offset.zero_offset()) 
        if s0 is None and 1 == self.vecdim:
            second_size = self.dims[1]*second_t_size
            first_off_base = sum([second_off_base for i in range(second_size)],lsc_offset.zero_offset()) 

        
        result = sum([first_off_base for i in range(first)],lsc_offset.zero_offset()) + \
                 sum([second_off_base for i in range(second)],lsc_offset.zero_offset())
            

        #print(f"mapped {idx} onto {result} (strides: {self.stride_indices})")
        return result
