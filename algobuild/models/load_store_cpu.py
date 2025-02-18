from abc import abstractmethod
from algobuild.generators import mm_op,tile,dimension_type

from algobuild.components.operation import operation
from algobuild.components import scalar_tile

class tile_offset_mapper:
    @abstractmethod
    def __call__(
            tile : tile,
            idx : (int,int)) -> int:
# I don't think we need the subindex for data origins
#            subidx : int) -> int:
        raise NotImplementedError("tried calling abstract tile offset mapper")

class lsc_operation:
    def __init__(self,
                 tiles : list[tile],
                 indices: list[list[int]],
                 reads: list[int],
                 writes: list[int]):
        self.tiles = tiles
        self.indices = indices
        self.reads = reads
        self.writes = writes

class lsc_load(lsc_operation):
    def __init__(self, rtype_idx : int, res_idx : int, addr_idx : int, off : int, t : tile):
        self.off = off

        tiles = [scalar_tile, t]
        indices = [[rtype_idx, addr_idx], [rtype_idx, res_idx]]
        # read address
        reads = [0]
        # write resource
        writes = [1]

        super(lsc_load, self).__init__(tiles=tiles, indices=indices, reads=reads, writes=writes)

    @property
    def rtype_idx(self):
        return self.indices[0][0]

    @property
    def res_idx(self):
        return self.indices[1][1]

    @property
    def addr_idx(self):
        return self.indices[0][1]

    @property
    def t(self):
        return self.tiles[1]

    def __str__(self):
        reg_chars = ['a','b','c']
        i = self.rtype_idx
        vladims = sum([1 if (d.dt == dimension_type.vla) else 0 for d in [self.t.dima,self.t.dimb] ])
        vlenstr = ""
        if 0 < vladims:
            vlenstr = "*VLEN"*vladims
        return f"{reg_chars[i]}{self.res_idx} <- LOAD {reg_chars[i]}a{self.addr_idx} + {self.off}{vlenstr}"

class lsc_zero(lsc_operation):
    def __init__(self, rtype_idx : int, res_idx : int, t : tile):

        tiles = [t]
        indices = [[rtype_idx, res_idx]]
        reads = []
        # write resource
        writes = [0]

        super(lsc_zero, self).__init__(tiles=tiles, indices=indices, reads=reads, writes=writes)
    @property
    def rtype_idx(self):
        return self.indices[0][0]

    @property
    def res_idx(self):
        return self.indices[0][1]

    @property
    def t(self):
        return self.tiles[0]

    def __str__(self):
        reg_chars = ['a','b','c']
        return f"{reg_chars[self.rtype_idx]}{self.res_idx} <- 0"

class lsc_addr_add(lsc_operation):
    def __init__(self, rtype_idx : int, addr_idx : int, off : int, t : tile):
        self.off = off

        # t is used for calculating address, isn't the tile the address resides in
        tiles = [scalar_tile, t]
        indices = [[rtype_idx, addr_idx]]
        # read addr
        reads = [0]
        # write addr
        writes = [0]

        super(lsc_addr_add, self).__init__(tiles=tiles, indices=indices, reads=reads, writes=writes)

    @property
    def rtype_idx(self):
        return self.indices[0][0]

    @property
    def addr_idx(self):
        return self.indices[0][1]

    @property
    def t(self):
        return self.tiles[1]

    def __str__(self):
        reg_chars = ['a','b','c']
        vladims = sum([1 if (d.dt == dimension_type.vla) else 0 for d in [self.t.dima,self.t.dimb] ])
        vlenstr = ""
        if 0 < vladims:
            vlenstr = "*VLEN"*vladims
        ar_str = f"{reg_chars[self.rtype_idx]}a{self.addr_idx}"
        return f"{ar_str} <- {ar_str} + {self.off}{vlenstr}"

class lsc_debugmsg(lsc_operation):
    def __init__(self, msg : str):
        super(lsc_debugmsg, self).__init__(tiles=tiles, indices=indices, reads=reads, writes=writes)
        self.msg = msg
    def __str__(self):
        return self.msg
    
class lsc_transformation(lsc_operation):
    def __init__(self, op : str,
                 res_indices : list[int],
                 sub_indices : list[int],
                 tiles : list[tile],
                 reads : list[int] = [0,1,2],
                 writes : list[int] = [2]):
        self.op = op
        indices = [[i,r,s] for i,(r,s) in enumerate(zip(res_indices,sub_indices))]
        super(lsc_transformation, self).__init__(tiles=tiles, indices=indices, reads=reads, writes=writes)

    @property
    def res_indices(self):
        return [idxlist[1] for idxlist in self.indices]

    @property
    def sub_indices(self):
        return [idxlist[2] for idxlist in self.indices]

    def __str__(self):
        reg_chars = ['a','b','c']
        #TODO: subindices
        out = f"{reg_chars[-1]}{self.res_indices[-1]} <- {self.op}("
        out += ", ".join([f"{reg_chars[i]}{r}" for i,r in enumerate(self.res_indices)])
        out += ")"
        return out

class load_store_cpu:
    def __init__(self,
                 res_counts : list[int],
                 res_steps : list[int],
                 addr_counts : list[int],
                 addr_offset_ranges : list[list[tuple[int,int]]],
                 addr_starts : list[list[int]],
                 preload_counts : list[int],
                 offset_mappers : list[tile_offset_mapper],
                 op : str = "fma"):

        for rc,pc in zip(res_counts,preload_counts):
            assert pc <= rc, "Can't preload more than max. specified number of resources"

        self.res_counts = res_counts
        self.res_steps = res_steps
        self.preload_counts = preload_counts
        self.offset_mappers = offset_mappers
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
                    for res_count,pre_count,idx,subidx in\
                    zip(self.res_counts,self.preload_counts,
                        self.res_indices,self.res_subindices)
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
        self.addr_offsets = [0]*len(self.res_counts)


        # Track last tile used per res to determine offset to next tile given
        # last offset used to load data for this res
        # NOTE: This is currently only used by preload(add_current_offsets=True)
        self.last_tile_used = [None]*len(self.res_counts)

    def is_in_range(self, current_offset : int, target_offset : int, offset_range : tuple[int,int]) -> bool:
        difference = target_offset - current_offset

        return (difference >= offset_range[0]) and (difference <= offset_range[1])

    def resolve_data(self, t : tile, res_idx : int, toff : int, rtype_idx : int) -> list[lsc_load|lsc_addr_add]:
        # check if the required data is already in the resource
        reg_chars = ["a","b","c"]
        corg = self.cdos[rtype_idx][res_idx]
        result = []
        if corg == -2:
            self.cdos[rtype_idx][res_idx] = toff
            corg = toff
        if corg == toff:
            return result

        vladims = sum([1 if (d.dt == dimension_type.vla) else 0 for d in [t.dima,t.dimb] ])
        vlenstr = ""
        if 0 < vladims:
            vlenstr = "*VLEN"*vladims


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
                    self.addr_offsets[rtype_idx] = 0
                    self.addr_reg_offsets[rtype_idx][addr_idx] = toff
                    break
                if self.is_in_range(current_offset=caoff,
                               target_offset=toff,
                               offset_range=offset_range):
                    off = toff-caoff
                    if off < offset_min:
                        offset_min = off
                        addr_idx_to_use = addr_idx
                        self.addr_offsets[rtype_idx] = off
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
            result.append(lsc_addr_add(rtype_idx=rtype_idx, addr_idx=addr_idx_to_increment, off=add_value, t=t))
            self.addr_reg_offsets[rtype_idx][addr_idx_to_increment] += add_value
        
        
        #ar_str = f"{reg_chars[rtype_idx]}a{addr_idx_to_use}"
        #res_str = f"{reg_chars[rtype_idx]}{res_idx}"

        self.cdos[rtype_idx][res_idx] = toff
        offpref = "o"
        if 0 < vladims:
            offpref = ""
        #result.append(f"{res_str} <- LOAD {ar_str} + {offpref}{self.addr_offsets[rtype_idx]}{vlenstr}")
        result.append(lsc_load(rtype_idx=rtype_idx, res_idx=res_idx,
                                    addr_idx=addr_idx_to_use,
                                    off=self.addr_offsets[rtype_idx],
                                    t=t))
        self.last_tile_used[rtype_idx] = t
        # advance the addr offset by the size of the data. This is important when updating indices
        # at the end of the preload/main loop
        self.addr_offsets[rtype_idx] += t.dima.size*t.dimb.size


        return result

        
    def map_one(self,
                a_tile : tile, b_tile : tile, c_tile : tile,
                a_idx : (int,int), b_idx : (int,int), c_idx : (int,int),
                m_subidx : int, n_subidx : int, k_subidx : int) -> list[lsc_operation]:

        reg_chars = ['a','b','c']
        tiles = [a_tile,b_tile,c_tile]
        indices = [a_idx,b_idx,c_idx]
        target_offsets = [ mapper(tile,idx) for  mapper,tile,idx in\
                zip(self.offset_mappers,tiles,indices)]

        result = []
        res_indices = []




        for i,(toff,dos,res_count,t) in enumerate(
                zip(target_offsets,self.cdos,self.res_counts,[a_tile,b_tile,c_tile])):
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

            res_indices.append(res_idx)
            #result.append(lsc_debugmsg(f"Need {toff} in {reg_chars[i]}{res_idx}"))
            result.extend(self.resolve_data(t=t,res_idx=res_idx, toff=toff, rtype_idx=i))

        areg = res_indices[0]
        breg = res_indices[1]
        creg = res_indices[2]
        #result.append(f"c{creg} <- {self.op}(a{areg},b{breg},c{creg})")
        result.append(lsc_transformation(op=self.op, res_indices=res_indices,
                                  sub_indices=[m_subidx,n_subidx,k_subidx],
                                  tiles=[a_tile,b_tile,c_tile]))
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

        initial_cdos = self.cdos.copy()
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
        preload_addr_reg_last_used_offsets = [[None]*count for count in self.addr_counts]
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
                    if op.rtype_idx not in zero_dims:
                        subresults.append(op)
                    caoff = self.addr_reg_offsets[op.rtype_idx][op.addr_idx]
                    preload_addr_reg_offsets[op.rtype_idx][op.addr_idx] = caoff + op.off
                    preload_addr_reg_last_used_tile[op.rtype_idx][op.addr_idx] = op.t
                if isinstance(op, lsc_load):
                    if op.rtype_idx in zero_dims:
                        subresults.append(lsc_zero(
                            rtype_idx=op.rtype_idx, res_idx=op.res_idx, t=op.t))
                    else:
                        subresults.append(op)
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

                # reached required number of preloads, copy state
                #if self.preload_counts[op.rtype_idx] <= preloads_done[op.rtype_idx]:
                #    preload_addr_reg_offsets[op.rtype_idx] = \
                #        self.addr_reg_offsets[op.rtype_idx].copy()
            # Debug preloads:
            #print(", ".join(f"{d}/{p}" for d,p in zip(preloads_done,self.preload_counts)))

            results.extend(subresults)
            i+= 1
        if update_addrs:
            for i,(addr_count,off) in enumerate(zip(self.addr_counts,self.addr_offsets)):
                # No address updates if data was just zeroed out
                if i in zero_dims:
                    continue
                # assumption: caoff_max+off+start are the correct offsets for the preloads
                max_idx = 0
                caoff_max = preload_addr_reg_offsets[i][max_idx]
                for addr_idx,caoff in preload_addr_reg_offsets[i].items():
                    if caoff > caoff_max:
                        caoff_max = caoff
                        max_idx = addr_idx
                off = preload_addr_reg_last_used_offsets[i][max_idx]
                t = preload_addr_reg_last_used_tile[i][max_idx]
                for addr_idx,start in zip(range(addr_count),self.addr_starts[i]):
                    caoff = preload_addr_reg_offsets[i][addr_idx]
                    add_off = caoff_max-caoff+off+t.dima.size*t.dimb.size+start
                    results.append(lsc_addr_add(rtype_idx=i, addr_idx=addr_idx, off=add_off, t=t))
                    preload_addr_reg_offsets[i][addr_idx] = caoff+add_off
        # update the internal states so the main block would be correct:
        self.addr_reg_offsets = [d.copy() for d in preload_addr_reg_offsets]
        # use the original dos and assign the preloaded data
        self.cdos = initial_cdos
        for i,dos in enumerate(preload_dos):
            for res_idx,orig in dos.items():
                self.cdos[i][res_idx] = orig
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
