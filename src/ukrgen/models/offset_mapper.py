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
                 flip_tile_dims : bool = False):
        self.dims = dims
        self.stride_indices = stride_indices
        self.flip_tile_dims = flip_tile_dims


    def get_ldst_size(self, t : tile) -> lsc_offset:
        # TODO: When we start implementing arbitrary vectorization directions,
        #       This will probably need a direction parameter
        #       Alternatively this could be handled in some other manner so that
        #       always using the first component in the mapper is correct
        return self.map_tile_idx(t=t, idx=(1,0))

    def map_tile_idx(self, t : tile, idx : tuple[int,int]) -> lsc_offset:
        if self.flip_tile_dims:
            first = t.dimb.size*self.dims[0]*idx[0]
            second = t.dima.size*self.dims[1]*idx[1]
        else:
            first = t.dima.size*self.dims[0]*idx[0]
            second = t.dimb.size*self.dims[1]*idx[1]

        result = lsc_offset.zero_offset()

        s0 = self.stride_indices[0]
        s1 = self.stride_indices[1]

        istile = (t.dima.dt == dimension_type.vla) and \
                 (t.dimb.dt == dimension_type.vla)

        isvector = (t.dima.dt == dimension_type.vla) != \
                   (t.dimb.dt == dimension_type.vla)

        isscalar = (t.dima.dt != dimension_type.vla) and \
                   (t.dimb.dt != dimension_type.vla)

        if istile:
            strides = set([s for s in [s0,s1] if s is not None])
            result += lsc_offset(
               {
                   stridexvlen(strides, {0,1}) : first*second
               },[],[],0)

        for stride, value, dimt in zip([s0,s1],
                                       [first,second],
                                       [t.dima.dt,t.dimb.dt]):
            if stride is not None:
                # TODO: evaluate if a simple if dimt vla/ else is enough
                if isvector:
                    if dimt == dimension_type.vla:
                        result += lsc_offset(
                               {
                                   stridexvlen(set([stride]), set([0])) : value
                               },[],[],0)
                    else:
                        stridelist = [0 for i in range(stride+1)]
                        stridelist[stride] = value
                        result += lsc_offset({},stridelist,[],0)
                elif isscalar:
                    #list at least as big as this index and
                    # set the value with that index
                    stridelist = [0 for i in range(stride+1)]
                    stridelist[stride] = value
                    result += lsc_offset({},stridelist,[],0)
            else:
                if isvector:
                       result += lsc_offset({},[],[value],0)
                elif isscalar:
                       result += lsc_offset({},[],[],value)

        #print(f"mapped {idx} onto {result} (strides: {self.stride_indices})")
        return result
