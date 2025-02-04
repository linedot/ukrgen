from abc import abstractmethod

from algobuild.generators import mm_op,tile,dimension_type

class dummy_mm_mapper:
    def __init__(self, op : str = "fma"):
        self.op = op
    def __call__(self, ops : list[mm_op]) -> list[str]:

        result = []
        idxstr = lambda dima,dimb,sidx : "" if dima.size==dimb.size else f".el[{sidx}]" if dima.size > dimb.size else f"+{sidx}"
        vlensuf = lambda d : "*VLEN" if d.dt == dimension_type.vla else ""
        for op in ops:

            a_idx_str = (f"({op.a_idx[0]}{vlensuf(op.a_tile.dima)}{idxstr(op.a_tile.dima,op.c_tile.dima,op.m_subidx)},"
                          f"{op.a_idx[1]}{vlensuf(op.a_tile.dimb)}{idxstr(op.a_tile.dimb,op.b_tile.dima,op.k_subidx)})")
            b_idx_str = (f"({op.b_idx[0]}{vlensuf(op.b_tile.dima)}{idxstr(op.b_tile.dima,op.a_tile.dimb,op.k_subidx)},"
                          f"{op.b_idx[1]}{vlensuf(op.b_tile.dimb)}{idxstr(op.b_tile.dimb,op.c_tile.dimb,op.n_subidx)})")
            c_idx_str = (f"({op.c_idx[0]}{vlensuf(op.c_tile.dima)}{idxstr(op.c_tile.dima,op.a_tile.dima,op.m_subidx)},"
                          f"{op.c_idx[1]}{vlensuf(op.c_tile.dimb)}{idxstr(op.c_tile.dimb,op.b_tile.dimb,op.n_subidx)})")
            result.append(
                f"C{c_idx_str} <- {self.op}(A{a_idx_str},B{b_idx_str},C{c_idx_str})"
            )
        return result

class tile_offset_mapper:
    @abstractmethod
    def __call__(
            tile : tile,
            idx : (int,int)) -> int:
# I don't think we need the subindex for data origins
#            subidx : int) -> int:
        raise NotImplementedError("tried calling abstract tile offset mapper")


class vn_mm_mapper:
    def __init__(self,
                 res_counts : list[int],
                 res_steps : list[int],
                 addr_counts : list[int],
                 addr_offset_ranges : list[list[tuple[int,int]]],
                 addr_starts : list[list[int]],
                 preload_counts : list[int],
                 mappers : list[tile_offset_mapper],
                 op : str = "fma"):

        for rc,pc in zip(res_counts,preload_counts):
            assert pc <= rc, "Can't preload more than max. specified number of resources"

        self.res_counts = res_counts
        self.res_steps = res_steps
        self.preload_counts = preload_counts
        self.mappers = mappers
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

    def is_in_range(self, current_offset : int, target_offset : int, offset_range : tuple[int,int]) -> bool:
        difference = target_offset - current_offset

        return (difference >= offset_range[0]) and (difference <= offset_range[1])

    def resolve_data(self, res_idx : int, toff : int, rtype_idx : int, vladims : int = 0) -> list[str]:
        # check if the required data is already in the resource
        reg_chars = ["a","b","c"]
        corg = self.cdos[rtype_idx][res_idx]
        if corg == -2:
            self.cdos[rtype_idx][res_idx] = toff
            corg = toff
        if corg == toff:
            return []

        vlenstr = ""
        if 0 < vladims:
            vlenstr = "*VLEN"*vladims

        result = []

        caoffs = self.addr_reg_offsets[rtype_idx]

        addr_idx_to_use = -1
        while -1 == addr_idx_to_use:
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
                    addr_idx_to_use = addr_idx
                    self.addr_offsets[rtype_idx] = toff-caoff
                    break
            if -1 != addr_idx_to_use:
                break

            # First address register
            addr_idx_to_increment = 0
            caoff_max = self.addr_reg_offsets[rtype_idx][addr_idx_to_increment]
            # if none of the registers are in range, increment the addr register with the highest value below target
            for addr_idx in range(self.addr_counts[rtype_idx]):
                caoff = self.addr_reg_offsets[rtype_idx][addr_idx]
                if caoff_max < caoff and caoff < toff:
                    addr_idx_to_increment = addr_idx
                    caoff_max = caoff

            # add first value of the ranges (i.e. if range is (-16,15), subtract 16 so that
            # target offset can be reached with the lowest offset value in the range)
            add_value = toff - caoff_max + self.addr_offset_ranges[rtype_idx][addr_idx_to_increment][0]
            ar_str = f"{reg_chars[rtype_idx]}a{addr_idx_to_increment}"
            result.append(f"{ar_str} <- {ar_str} + {add_value}{vlenstr}")
            self.addr_reg_offsets[rtype_idx][addr_idx_to_increment] += add_value
        
        
        ar_str = f"{reg_chars[rtype_idx]}a{addr_idx_to_use}"
        res_str = f"{reg_chars[rtype_idx]}{res_idx}"

        self.cdos[rtype_idx][res_idx] = toff
        offpref = "o"
        if 0 < vladims:
            offpref = ""
        result.append(f"{res_str} <- LOAD {ar_str} + {offpref}{self.addr_offsets[rtype_idx]}{vlenstr}")

        return result

        
    def map_one(self,
                a_tile : tile, b_tile : tile, c_tile : tile,
                a_idx : (int,int), b_idx : (int,int), c_idx : (int,int),
                m_subidx : int, n_subidx : int, k_subidx : int) -> list[str]:

        target_offsets = [ mapper(tile,idx) for  mapper,tile,idx in\
                zip(self.mappers,[a_tile,b_tile,c_tile],[a_idx,b_idx,c_idx])]

        result = []
        res_indices = []

        # This seems "too clever", basically if a tile is vla,fixed, vladims=1, 
        # if it's vla,vla, vladims = 2
        off_vladims = [1 if (d.dt == dimension_type.vla) else 0 for t in [a_tile,b_tile,c_tile] for d in [t.dima,t.dimb] ]
        num_dims = 2
        off_vladims = [sum(off_vladims[i:i+num_dims]) for i in range(0, len(off_vladims), num_dims)]

        for i,(toff,dos,res_count) in enumerate(
                zip(target_offsets,self.cdos,self.res_counts)):
            res_idx = -1

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
            #print(f"Data change o{corg} -> o{toff*ofac} for dreg {suf}{reg} required")
            result.extend(self.resolve_data(res_idx=res_idx, toff=toff, rtype_idx=i, vladims=off_vladims[i]))

        areg = res_indices[0]
        breg = res_indices[1]
        creg = res_indices[2]
        result.append(f"c{creg} <- {self.op}(a{areg},b{breg},c{creg})")
        return result

    def __call__(self, ops : list[mm_op]) -> list[str]:
        result = []
        for op in ops:
            result.extend(self.map_one(
                a_tile = op.a_tile, b_tile = op.b_tile, c_tile = op.c_tile,
                a_idx = op.a_idx, b_idx = op.b_idx, c_idx = op.c_idx,
                m_subidx = op.m_subidx, n_subidx = op.n_subidx, k_subidx = op.k_subidx
                ))
        return result
