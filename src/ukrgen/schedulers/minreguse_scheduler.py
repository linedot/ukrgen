from copy import deepcopy

from ..models.load_store_operations import lsc_operation, lsc_reg_type
from ..models.load_store_cpu import lsc_state
from ..models.lsc.index import lsc_reg_index
from ..specializers.asm import lsc_specializer
from ..components.tile import determine_dreg_tag

class minreguse_scheduler:
    """
    Scheduler for minimizing number of data registers that need to be allocated
    by reusing already allocated registers that are no longer needed
    """
    def __init__(self, 
                 inflight_target : int = 1):

        self.allocated_registers = dict()
        self.free_allocated_registers = dict()
        self.new_registers = dict()

    def register_exists(self, index : lsc_reg_index, dreg_tag : str):
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
                    self.register_exists(op.indices[idx],dreg_tag)

    def register_to_read(self, index: lsc_reg_index, dreg_tag : str):
        if not index in self.allocated_registers[dreg_tag]:
            print(f"registering to alloc reg: {index}")
            if dreg_tag not in self.new_registers:
                self.new_registers[dreg_tag] = set()
            self.new_registers[dreg_tag].add(index)
        else:
            print(f"removing from free list: {index}")
            self.free_allocated_registers[dreg_tag].remove(index)

    def register_to_write(self, index: lsc_reg_index, dreg_tag : str):
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
                    self.register_to_read(op.indices[idx], dreg_tag)
            for idx in op.writes:
                if lsc_reg_type.data == op.reg_types[idx]:
                    dreg_tag = determine_dreg_tag(
                            op.tiles[idx].dima, op.tiles[idx].dimb)
                    self.register_to_write(op.indices[idx], dreg_tag)

    def reorder(self, ops: list[lsc_operation]) -> list[lsc_operation]:
        return ops

    def replace(self, ops: list[lsc_operation], data_registers) -> list[lsc_operation]:


        print(f"{self.new_registers}")
        print(f"{self.free_allocated_registers}")

        last_reads = dict()
        last_writes = dict()

        def determine_last_rw(op_idx : int,
                              op : lsc_operation,
                              idxindices : list[int], 
                              usage_dict : dict[str,dict[lsc_reg_index,int]]):
            for i in idxindices:
                idx = op.indices[i]
                if lsc_reg_type.data == op.reg_types[i]:
                    dreg_tag = determine_dreg_tag(
                            op.tiles[i].dima,
                            op.tiles[i].dimb)
                    if dreg_tag not in usage_dict:
                        usage_dict[dreg_tag] = {idx : op_idx}
                    else:
                        usage_dict[dreg_tag][idx] = op_idx


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
                if idx in hard_used:
                    continue
                if lsc_reg_type.data == op.reg_types[i]:
                    dreg_tag = determine_dreg_tag(
                            op.tiles[i].dima, op.tiles[i].dimb)

                    old_idx = idx
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
                    if idx in replace_dict[dreg_tag]:
                        print(f"replacing: {idx} with {replace_dict[dreg_tag][idx]}")
                        op.indices[i] = replace_dict[dreg_tag][idx]
                    
                    print(f"Adding index to used indices this op:{op.indices[i]}")
                    indices_this_op.append(op.indices[i])
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
            for lri in replace_dict[dreg_tag].keys():
                if lri not in self.allocated_registers[dreg_tag]:
                    data_registers[lri.component].remove(lri.indices[0])

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
