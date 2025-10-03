from copy import deepcopy

from ..models.load_store_operations import lsc_operation, lsc_reg_type
from ..models.load_store_cpu import lsc_state
from ..specializers.asm import lsc_specializer
from ..components.tile import determine_dreg_tag




# TODO: index wrapper and operations here probably mean that the index in lsc
#       operations should be a class with those methods as members

# TODO: This should be move to load_store_operations.py and used
#       more broadly in places where a conversion to a string is necessary
def index_to_str(index : tuple[str,list[int]]) -> str:
    """
    Converts an LSC register index to a string
    """
    iicount = len(index[1])
    # TODO: is iicount > 2 even valid? Maybe some kind of fixed size tile
    #       register with 2D subindexing?
    if iicount > 2 or iicount < 1:
        raise ValueError(f"Invalid number of numeric indices in index: {iicount}")
    if iicount == 2:
        return f"{index[0]}{index[1][0]}.el{index[1][1]}"
    if iicount == 1:
        return f"{index[0]}{index[1][0]}"

class mru_index:
    """
    LSC index wrapper that can be used as a dict key or an element in a set
    """
    def __init__(self, index : tuple[str,list[int]]):
        self.index = deepcopy(index)

    def __eq__(self, other):
        return self.index[0] == other.index[0] and \
                  all([i1 == i2 for i1,i2 in zip(
                      self.index[1], other.index[1]
                      )])

    def __str__(self) -> str:
        return index_to_str(self.index)

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash(str(self))

class minreguse_scheduler:
    """
    Scheduler for minimizing number of data registers that need to be allocated
    by reusing already allocated registers that are no longer needed
    """
    def __init__(self):

        self.allocated_registers = dict()
        self.free_allocated_registers = dict()
        self.new_registers = dict()

    def register_exists(self, index : mru_index, dreg_tag : str):
        if dreg_tag not in self.allocated_registers:
            self.allocated_registers[dreg_tag] = set()
        print(f"registering existing reg: {index}")
        self.allocated_registers[dreg_tag].add(index)

    def analyze_preceeding(self, ops: list[lsc_operation]):
        for op in ops:
            for idx in op.reads+op.writes:
                if lsc_reg_type.data == op.reg_types[idx]:
                    dreg_tag = determine_dreg_tag(
                            op.tiles[idx].dima, op.tiles[idx].dimb)
                    self.register_exists(mru_index(op.indices[idx]),dreg_tag)

    def register_to_read(self, index: mru_index, dreg_tag : str):
        if not index in self.allocated_registers[dreg_tag]:
            print(f"registering to alloc reg: {index}")
            if dreg_tag not in self.new_registers:
                self.new_registers[dreg_tag] = set()
            self.new_registers[dreg_tag].add(index)
        else:
            print(f"removing from free list: {index}")
            self.free_allocated_registers[dreg_tag].remove(index)

    def register_to_write(self, index: mru_index, dreg_tag : str):
        if not index in self.allocated_registers[dreg_tag]:
            print(f"registering to alloc reg: {index}")
            if dreg_tag not in self.new_registers:
                self.new_registers[dreg_tag] = set()
            self.new_registers[dreg_tag].add(index)
        else:
            print(f"removing from free list: {index}")
            self.free_allocated_registers[dreg_tag].remove(index)

    def analyze(self, ops: list[lsc_operation]):
        self.free_allocated_registers = deepcopy(self.allocated_registers)
        for op in ops:
            for idx in op.reads:
                if lsc_reg_type.data == op.reg_types[idx]:
                    op.tiles[idx].dima
                    dreg_tag = determine_dreg_tag(
                            op.tiles[idx].dima, op.tiles[idx].dimb)
                    self.register_to_read(mru_index(op.indices[idx]), dreg_tag)
            for idx in op.writes:
                if lsc_reg_type.data == op.reg_types[idx]:
                    dreg_tag = determine_dreg_tag(
                            op.tiles[idx].dima, op.tiles[idx].dimb)
                    self.register_to_write(mru_index(op.indices[idx]), dreg_tag)

    def replace(self, ops: list[lsc_operation], data_registers) -> list[lsc_operation]:


        print(f"{self.new_registers}")
        print(f"{self.free_allocated_registers}")

        last_reads = dict()
        last_writes = dict()

        def determine_last_rw(op_idx : int,
                              op : lsc_operation,
                              idxindices : list[int], 
                              usage_dict : dict[str,dict[mru_index,int]]):
            for i in idxindices:
                idx = op.indices[i]
                if lsc_reg_type.data == op.reg_types[i]:
                    dreg_tag = determine_dreg_tag(
                            op.tiles[i].dima,
                            op.tiles[i].dimb)
                    if dreg_tag not in usage_dict:
                        usage_dict[dreg_tag] = {mru_index(idx) : op_idx}
                    else:
                        usage_dict[dreg_tag][mru_index(idx)] = op_idx


        for op_idx,op in enumerate(ops):
            determine_last_rw(op_idx, op, op.reads, last_reads)
            determine_last_rw(op_idx, op, op.writes, last_writes)

        replace_dict = { dreg_tag : dict() for
                         dreg_tag in self.free_allocated_registers.keys() }


        hard_used = set()

        result_ops = []
        for op_idx,op in enumerate(ops):
            print(f"processing OP {op_idx}: {op}")

            indices_this_op = []
            tiles_this_op = []


            for i in op.writes:
                idx = op.indices[i]
                if mru_index(idx) in hard_used:
                    continue
                if lsc_reg_type.data == op.reg_types[i]:
                    dreg_tag = determine_dreg_tag(
                            op.tiles[i].dima, op.tiles[i].dimb)

                    old_idx = mru_index(idx)
                    if old_idx not in replace_dict[dreg_tag]:
                        if self.free_allocated_registers[dreg_tag]:
                            rpl_idx = self.free_allocated_registers[dreg_tag].pop()
                            replace_dict[dreg_tag][old_idx] = rpl_idx

                            print(f"Setting last read of {rpl_idx} to last read of {old_idx} : {last_reads[dreg_tag][old_idx]}")
                            last_reads[dreg_tag][rpl_idx] = \
                                    last_reads[dreg_tag][old_idx]
                            print(f"Setting last write of {rpl_idx} to last write of {old_idx} : {last_writes[dreg_tag][old_idx]}")
                            last_writes[dreg_tag][rpl_idx] = \
                                    last_writes[dreg_tag][old_idx]


            for i,idx in enumerate(op.indices):
                if lsc_reg_type.data == op.reg_types[i]:
                    dreg_tag = determine_dreg_tag(
                            op.tiles[i].dima, op.tiles[i].dimb)


                    # replace if in map
                    if mru_index(idx) in replace_dict[dreg_tag]:
                        print(f"replacing: {mru_index(idx)} with {replace_dict[dreg_tag][mru_index(idx)]}")
                        op.indices[i] = replace_dict[dreg_tag][mru_index(idx)].index
                    
                    print(f"Adding index to used indices this op:{mru_index(op.indices[i])}")
                    indices_this_op.append(mru_index(op.indices[i]))
                    tiles_this_op.append(op.tiles[i])

            for used_idx,used_tile in zip(indices_this_op,tiles_this_op):
                dreg_tag = determine_dreg_tag(
                        used_tile.dima,
                        used_tile.dimb)
                if op_idx == last_reads[dreg_tag][used_idx]:
                    print(f"Last read of {used_idx}, releasing it")
                    self.free_allocated_registers[dreg_tag].add(used_idx)

            
            hard_used = hard_used.union(set(indices_this_op))

            result_ops.append(deepcopy(op))

        for dreg_tag in self.free_allocated_registers.keys():
            for mrui in replace_dict[dreg_tag].keys():
                if mrui not in self.allocated_registers[dreg_tag]:
                    data_registers[mrui.index[0]].remove(mrui.index[1][0])

        return result_ops

#    def reschedule(self, ops : list[lsc_operation]) -> list[lsc_operation]:
#
#        #analyse
#        result = []
#        position = 0
#        for op in ops:
#            for idx in op.reads:
#                if lsc_reg_type.data == op.reg_types[idx]:
#                    sel
#            for idx in op.writes:
#                if lsc_reg_type.data == op.reg_types[idx]:
#                    self.last_uses[index_to_str(op.indices[idx])] = positionf.last_uses[index_to_str(op.indices[idx])] = position
#
#            position += 1


        # reschedule
