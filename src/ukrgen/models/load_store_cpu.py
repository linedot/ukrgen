import copy
from abc import abstractmethod
from enum import Enum,auto

from ..generators import mm_op,tile,dimension_type
from ..components.operation import operation
from ..components import scalar_tile

from .load_store_operations import (
    lsc_addr_add,
    lsc_debugmsg,
    lsc_load,
    lsc_operation,
    lsc_store,
    lsc_transformation,
    lsc_zero,
)

class lsc_state(Enum):
    clean = auto()
    loaded = auto()
    modified = auto()


class tile_offset_mapper:
    @abstractmethod
    def __call__(
            tile : tile,
            idx : (int,int)) -> int:
# I don't think we need the subindex for data origins
#            subidx : int) -> int:
        raise NotImplementedError("tried calling abstract tile offset mapper")


class load_store_cpu:
    def __init__(self,
                 res_counts : list[int],
                 res_steps : list[int],
                 addr_counts : list[int],
                 addr_offset_ranges : list[list[tuple[int,int]]],
                 addr_starts : list[list[int]],
                 preload_counts : list[int],
                 offset_mappers : list[tile_offset_mapper],
                 resolve_order : list[int] = [0,1,2],
                 op : str = "fma"):

        for rc,pc in zip(res_counts,preload_counts):
            assert pc <= rc, "Can't preload more than max. specified number of resources"

        self.res_counts = res_counts
        self.res_steps = res_steps
        self.preload_counts = preload_counts
        self.offset_mappers = offset_mappers
        self.resolve_order = resolve_order
        self.op = op

        self.addr_counts = addr_counts
        self.addr_starts = addr_starts
        self.addr_offset_ranges = addr_offset_ranges

        self.reset()

    def reset(self):

        self.res_indices = [0]*len(self.res_counts)
        self.res_subindices = [0]*len(self.res_counts)

        self.addr_indices = [0]*len(self.addr_counts)

        # For now: -1 = invalid data, replace origin, schedule load
        #          -2 = data preloaded, replace origin, but don't schedule a load
        # TODO: revisit with not-magic-number based approach?
        self.cdos = [
            { i : -2 if i < pre_count else -1 for i in range(res_count)}\
                    for res_count,pre_count in\
                    zip(self.res_counts,self.preload_counts)
        ]

        self.states = [
            { i : lsc_state.loaded if i < pre_count else lsc_state.clean for i in range(res_count)}\
                    for res_count,pre_count in\
                    zip(self.res_counts,self.preload_counts)
        ]

        # Offsets to base address corresponding to absolute address values 
        # currently stored in the address registers
        # -2 = assume correct value in addr, replace tracked value, but don't
        #      schedule update operation
        self.addr_reg_offsets = [
            { i : -2 if i < pre_count else starts[i] for i in range(addr_count)}\
                    for pre_count,starts,addr_count in\
                    zip(self.preload_counts,self.addr_starts,self.addr_counts)
        ]
        # ABC data offsets to addresses currently stored in the address registers
        self.load_offsets = [0]*len(self.res_counts)
        self.store_offsets = [0]*len(self.res_counts)


        # Track last tile used per res to determine offset to next tile given
        # last offset used to load data for this res
        # NOTE: This is currently only used by preload(add_current_offsets=True)
        self.last_tile_used = [None]*len(self.res_counts)

    def is_in_range(self, current_offset : int, target_offset : int, offset_range : tuple[int,int]) -> bool:
        difference = target_offset - current_offset

        return (difference >= offset_range[0]) and (difference <= offset_range[1])

    def resolve_addr(self, rtype_idx : int, toff : int, addr_offsets : list[int],
                     op_list : list[lsc_operation], t : tile) -> int:
        caoffs = self.addr_reg_offsets[rtype_idx]
        addr_idx_to_use = -1
        while -1 == addr_idx_to_use:
            # TODO: better starting value
            offset_min = 99999999
            # check if we can address the data with one of the address registers + immediate offset
            for addr_idx in range(self.addr_counts[rtype_idx]):
                caoff = caoffs[addr_idx]
                offset_range = self.addr_offset_ranges[rtype_idx][addr_idx]
                if (caoff == -2):
                    addr_idx_to_use = addr_idx
                    addr_offsets[rtype_idx] = 0
                    self.addr_reg_offsets[rtype_idx][addr_idx] = toff
                    break
                if self.is_in_range(current_offset=caoff,
                               target_offset=toff,
                               offset_range=offset_range):
                    off = toff-caoff
                    if off < offset_min:
                        offset_min = off
                        addr_idx_to_use = addr_idx
                        addr_offsets[rtype_idx] = off
            if -1 != addr_idx_to_use:
                break

            # First address register
            addr_idx_to_increment = 0
            caoff_min = self.addr_reg_offsets[rtype_idx][addr_idx_to_increment]
            # if none of the registers are in range, increment the addr register with the lowest value below target
            for addr_idx in range(self.addr_counts[rtype_idx]):
                caoff = self.addr_reg_offsets[rtype_idx][addr_idx]
                if caoff_min > caoff and caoff < toff:
                    addr_idx_to_increment = addr_idx
                    caoff_min = caoff

            # add first value of the ranges (i.e. if range is (-16,15), subtract 16 so that
            # target offset can be reached with the lowest offset value in the range)
            add_value = toff - caoff_min + self.addr_offset_ranges[rtype_idx][addr_idx_to_increment][0]
            #ar_str = f"{reg_chars[rtype_idx]}a{addr_idx_to_increment}"
            #result.append(f"{ar_str} <- {ar_str} + {add_value}{vlenstr}")
            op_list.append(lsc_addr_add(rtype_idx=rtype_idx, addr_idx=addr_idx_to_increment, off=add_value, t=t))
            self.addr_reg_offsets[rtype_idx][addr_idx_to_increment] += add_value

        return addr_idx_to_use

    def resolve_data(self, t : tile, res_idx : int, toff : int, rtype_idx : int) -> list[lsc_operation]:
        # check if the required data is already in the resource
        reg_chars = ["a","b","c"]
        corg = self.cdos[rtype_idx][res_idx]
        result = []
        if corg == -2:
            self.cdos[rtype_idx][res_idx] = toff
            corg = toff
        if corg == toff:
            return result

        # store if dirty
        if lsc_state.modified == self.states[rtype_idx][res_idx]:
            addr_idx_for_store = self.resolve_addr(rtype_idx=rtype_idx, toff=corg,
                                                   addr_offsets=self.store_offsets,
                                                   op_list=result, t=t)
            result.append(lsc_store(rtype_idx=rtype_idx, res_idx=res_idx,
                                    addr_idx=addr_idx_for_store,
                                    off=self.store_offsets[rtype_idx],
                                    t=t))

        # load
        addr_idx_for_load = self.resolve_addr(rtype_idx=rtype_idx, toff=toff,
                                              addr_offsets=self.load_offsets,
                                              op_list=result, t=t)
        self.cdos[rtype_idx][res_idx] = toff
        self.states[rtype_idx][res_idx] = lsc_state.loaded
        result.append(lsc_load(rtype_idx=rtype_idx, res_idx=res_idx,
                                    addr_idx=addr_idx_for_load,
                                    off=self.load_offsets[rtype_idx],
                                    t=t))
        self.last_tile_used[rtype_idx] = t

        # advance the addr offset by the size of the data. This is important when updating indices
        # at the end of the preload/main loop
        self.load_offsets[rtype_idx] += t.dima.size*t.dimb.size


        return result

    def store_modified(self) -> list[lsc_operation]:
        result = []
        for rtype_idx,res_count in enumerate(self.res_counts):
            for res_idx in range(res_count):
                if lsc_state.modified == self.states[rtype_idx][res_idx]:
                    corg = self.cdos[rtype_idx][res_idx]
                    # NOTE: are different tiles for different res indices feasible?
                    t = self.last_tile_used[rtype_idx]
                    addr_idx_for_store = self.resolve_addr(
                            rtype_idx=rtype_idx, toff=corg,
                            addr_offsets=self.store_offsets,
                            op_list=result, t=t)
                    result.append(lsc_store(rtype_idx=rtype_idx, res_idx=res_idx,
                                            addr_idx=addr_idx_for_store,
                                            off=self.store_offsets[rtype_idx],
                                            t=t))
                    self.states[rtype_idx][res_idx] = lsc_state.clean
        return result

        
    def map_one(self,
                a_tile : tile, b_tile : tile, c_tile : tile,
                a_idx : (int,int), b_idx : (int,int), c_idx : (int,int),
                m_subidx : int, n_subidx : int, k_subidx : int) -> list[lsc_operation]:

        reg_chars = ['a','b','c']
        tiles = [a_tile,b_tile,c_tile]
        a_subidx = None
        if a_tile.dima.size > c_tile.dima.size:
            a_subidx = m_subidx
            c_idx = (c_idx[0]+m_subidx, c_idx[1])
        if a_tile.dimb.size > b_tile.dima.size:
            a_subidx = k_subidx
            b_idx = (b_idx[0]+k_subidx, b_idx[1])

        b_subidx = None
        if b_tile.dima.size > a_tile.dimb.size:
            b_subidx = k_subidx
            a_idx = (a_idx[0], a_idx[1]+k_subidx)
        if b_tile.dimb.size > c_tile.dimb.size:
            b_subidx = n_subidx
            c_idx = (c_idx[0], c_idx[1]+n_subidx)
            

        c_subidx = None
        if c_tile.dima.size > a_tile.dima.size:
            c_subidx = m_subidx
            a_idx = (a_idx[0]+m_subidx, a_idx[1])
        if c_tile.dimb.size > b_tile.dimb.size:
            c_subidx = n_subidx
            b_idx = (b_idx[0], b_idx[1]+n_subidx)
        indices = [a_idx,b_idx,c_idx]
        target_offsets = [ mapper(tile,idx) for  mapper,tile,idx in\
                zip(self.offset_mappers,tiles,indices)]

        result = []
        res_indices = [-1]*len(self.res_counts)




        
        #for i,(toff,dos,res_count,t) in enumerate(
        #        zip(target_offsets,self.cdos,self.res_counts,[a_tile,b_tile,c_tile])):
        for i in self.resolve_order[:len(self.res_counts)]:
            toff = target_offsets[i]
            dos = self.cdos[i]
            res_count = self.res_counts[i]
            t = tiles[i]
            res_idx = -1

            #result.append(lsc_debugmsg(f"idx {i}:{indices[i]} translated to offset {toff}"))

            # check if any of the registers have the data
            for j in range(res_count):
                if (toff == dos[j]) or (-2 == dos[j]):
                    res_idx = j
                    break

            # use the current index for this input, use the next next time
            if res_idx == -1:
                res_idx = self.res_indices[i]
                self.res_indices[i] = (res_idx+self.res_steps[i]) % self.res_counts[i]

            res_indices[i] = res_idx
            #result.append(lsc_debugmsg(f"Need {toff} in {reg_chars[i]}{res_idx}"))
            result.extend(self.resolve_data(t=t,res_idx=res_idx, toff=toff, rtype_idx=i))

        areg = res_indices[0]
        breg = res_indices[1]
        creg = res_indices[2]
        #result.append(f"c{creg} <- {self.op}(a{areg},b{breg},c{creg})")


        self.states[2][creg] = lsc_state.modified
        result.append(lsc_transformation(op=self.op, res_indices=res_indices,
                                  sub_indices=[a_subidx,b_subidx,c_subidx],
                                  tiles=[a_tile,b_tile,c_tile]))
        self.last_tile_used[0] = a_tile
        self.last_tile_used[1] = b_tile
        self.last_tile_used[2] = c_tile
        return result

    def preload(self, ops : list[mm_op],
                zero_addrs : bool = False,
                update_addrs : bool = True,
                zero_dims : list[int] = [2],
                ignore_dims : list[int] = [],
                add_current_offsets : bool = False,
                ) -> list[lsc_operation]:
        results = []
        preloads_done = [0]*len(self.res_counts)
        # treat ignored as already preloaded
        for dim in ignore_dims:
            preloads_done[dim] = self.preload_counts[dim]

        pre_preload_states = copy.deepcopy(self.states)

        initial_cdos = copy.deepcopy(self.cdos)
        self.cdos = [{ i : -1 for i in range(res_count)} for res_count in self.res_counts]
        if zero_addrs:
            self.addr_reg_offsets = [
                { i : 0 for i in range(addr_count)}\
                        for addr_count in\
                        self.addr_counts
            ]
        #else:
        #    self.addr_reg_offsets = [
        #        { i : starts[i] for i in range(addr_count)}\
        #                for starts,addr_count in\
        #                zip(self.addr_starts,self.addr_counts)
        #    ]
        # TODO: decouple a reg tracker structure and use it instead of 
        #       tracking manually
        #results.append(lsc_debugmsg("\n  ".join(map(str,self.cdos))))

        # NOTE: Another pythonism, the commented out line and the next one
        #       are not equivalent and the former will result in partly
        #       overlapping dicts
        # preload_dos = [{}]*len(self.res_counts)
        preload_dos = [{} for i in range(len(self.res_counts))]
        # NOTE: gotta copy the dicts explicitly, otherwise they'll be references
        preload_addr_reg_offsets = [d.copy() for d in self.addr_reg_offsets]

        # initialize with currently saved values
        preload_addr_reg_last_used_offsets = \
            [[loff-1]*count for loff,count in zip(self.load_offsets,self.addr_counts)]

        preload_addr_reg_last_used_tile = [ [None]*count for count in self.addr_counts]
        i = 0
        reg_chars = ['a','b','c']
        while any([p > d for p,d in zip(self.preload_counts,preloads_done)]):
            op = ops[i]
            subresults = []
            mapresults = self.map_one(
                a_tile = op.a_tile, b_tile = op.b_tile, c_tile = op.c_tile,
                a_idx = op.a_idx, b_idx = op.b_idx, c_idx = op.c_idx,
                m_subidx = op.m_subidx, n_subidx = op.n_subidx, k_subidx = op.k_subidx
                )
            for op in mapresults:
                if isinstance(op, lsc_transformation):
                    continue
                if isinstance(op, lsc_debugmsg):
                    subresults.append(op)
                    continue
                if self.preload_counts[op.rtype_idx] <= preloads_done[op.rtype_idx]:
                    continue
                if isinstance(op, lsc_addr_add):
                    caoff = self.addr_reg_offsets[op.rtype_idx][op.addr_idx]
                    preload_addr_reg_offsets[op.rtype_idx][op.addr_idx] = caoff + op.off
                    preload_addr_reg_last_used_tile[op.rtype_idx][op.addr_idx] = op.t

                    if op.rtype_idx not in zero_dims:
                        subresults.append(op)
                if isinstance(op, lsc_load):
                    caoff = self.addr_reg_offsets[op.rtype_idx][op.addr_idx]
                    # Some updates to offsets are implicit, therefore update
                    preload_addr_reg_offsets[op.rtype_idx][op.addr_idx] = caoff
                    new_do = caoff + op.off
                    #subresults.append(lsc_debugmsg(f"{reg_chars[op.rtype_idx]}{op.res_idx} now holds {new_do}"))
                    preload_dos[op.rtype_idx][op.res_idx] = new_do
                    tsize = op.t.dima.size*op.t.dimb.size
                    preload_addr_reg_last_used_offsets[op.rtype_idx][op.addr_idx] = op.off
                    preload_addr_reg_last_used_tile[op.rtype_idx][op.addr_idx] = op.t

                    preloads_done[op.rtype_idx] += 1
                    if op.rtype_idx in zero_dims:
                        subresults.append(lsc_zero(
                            rtype_idx=op.rtype_idx, res_idx=op.res_idx, t=op.t))
                    else:
                        subresults.append(op)

            results.extend(subresults)
            i+= 1

        if update_addrs:
            for i,addr_count in enumerate(self.addr_counts):
                # No address updates if data was just zeroed out
                if i in zero_dims:
                    continue
                # assumption: caoff_max+off+start are the correct offsets for the preloads

                # Find the register holding the highest address, use that as new zero
                max_idx = 0
                caoff_max = preload_addr_reg_offsets[i][max_idx]
                for addr_idx,caoff in preload_addr_reg_offsets[i].items():
                    if caoff > caoff_max:
                        caoff_max = caoff
                        max_idx = addr_idx

                # the imm offset this address register has been used with
                off = preload_addr_reg_last_used_offsets[i][max_idx]

                # no loads happened, therefore no offset
                # 0 would mean a load happened with offset 0 further down,
                # so set to -1 to make (off+1)==0
                if None == off:
                    off = -1
                t = preload_addr_reg_last_used_tile[i][max_idx]

                # no loads happaned, therefor no tile used for a load
                # Tile will have been set for a tranform, use it
                if None == t:
                    t = self.last_tile_used[i]
                for addr_idx,start in zip(range(addr_count),self.addr_starts[i]):
                    caoff = preload_addr_reg_offsets[i][addr_idx]


                    # example:
                    # aa0 holds ptr + caoff 2, was used with imm off 3
                    # aa1 holds ptr + caoff 4, was used with imm off 3
                    # caoff_max = 4
                    # aa0.caoff = 2, aa1.caoff = 4
                    # aa0.off=aa1.off=3
                    # all tile sizes are 1
                    # aa0.start=0, aa1.start=2

                    add_off = caoff_max-caoff+(off+1)*t.dima.size*t.dimb.size+start

                    #aa0.add_off = 4-2+3*1+0 = 5
                    #aa1.add_off = 4-4+3*1+2 = 5

                    # After:
                    # aa0.caoff = 2+5 = 7
                    # aa1.caoff = 4+5 = 9

                    # print(f"rtype_idx: {i}")
                    # print(f"add_off: {add_off}, caoff_max: {caoff_max}, off: {off}")
                    # print(f"caoff: {caoff}")
                    # print(f"tsz: {t.dima.size,t.dimb.size}, start: {start}")

                    if add_off == 0:
                        continue
                    results.append(lsc_addr_add(rtype_idx=i, addr_idx=addr_idx, off=add_off, t=t))
                    preload_addr_reg_offsets[i][addr_idx] = caoff+add_off

        # This messes up zero_dims/ignore_dims stuff
        # self.addr_reg_offsets = [d.copy() for d in preload_addr_reg_offsets]
        # update the internal states so the main block would be correct:
        self.addr_reg_offsets = [
            { i : -2 if i < pre_count else starts[i] for i in range(addr_count)}\
                    for pre_count,starts,addr_count in\
                    zip(self.preload_counts,self.addr_starts,self.addr_counts)
        ]
        # use the original dos and assign the preloaded data
        self.cdos = copy.deepcopy(initial_cdos)
        for i,dos in enumerate(preload_dos):
            for res_idx,orig in dos.items():
                self.cdos[i][res_idx] = orig

        # correct load states were already set in self.reset(). preload shouldn't change it
        self.states = copy.deepcopy(pre_preload_states)
        #results.append(lsc_debugmsg("dos after preload:\n    " + "\n    ".join(map(str,self.cdos))))
        #results.append(lsc_debugmsg("res idx after preload: " + ", ".join(map(str,self.res_indices))))
        # reset the indices
        self.res_indices = [ d%count for d,count in zip(preloads_done,self.res_counts)]
        return results
    def __call__(self, ops : list[mm_op]) -> list[str]:
        result = []
        for op in ops:
            result.extend(self.map_one(
                a_tile = op.a_tile, b_tile = op.b_tile, c_tile = op.c_tile,
                a_idx = op.a_idx, b_idx = op.b_idx, c_idx = op.c_idx,
                m_subidx = op.m_subidx, n_subidx = op.n_subidx, k_subidx = op.k_subidx
                ))
        return result
