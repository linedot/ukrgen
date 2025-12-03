# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from copy import deepcopy
from collections import deque

from ..models.load_store_operations import (
        lsc_operation,
        lsc_debugmsg,
        lsc_reg_type as lrt,
        can_reorder
        )
from ..models.load_store_cpu import lsc_state
from ..models.lsc.index import lsc_reg_index
from ..models.lsc.reg import lsc_reg
from ..specializers.asm import lsc_specializer
from ..components.tile import determine_dreg_tag

class reg_usage_tracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.writes : dict[lsc_reg,set[int]] = dict()
        self.reads : dict[lsc_reg,set[int]] = dict()

    def add_read(self, reg : lsc_reg, pos : int):
        if reg not in self.reads:
            self.reads[reg] = set()

        self.reads[reg].add(pos)

    def del_read(self, reg : lsc_reg, pos : int):
        self.reads[reg].remove(pos)

    def add_write(self, reg : lsc_reg, pos : int):
        if reg not in self.writes:
            self.writes[reg] = set()

        self.writes[reg].add(pos)

    def del_write(self, reg : lsc_reg, pos : int):
        self.writes[reg].remove(pos)

    def last_write(self, reg : lsc_reg):
        return max(self.writes[reg])

    def last_read(self, reg : lsc_reg):
        return max(self.reads[reg])

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

        self.rut = reg_usage_tracker()


    def register_exists(self, reg : lsc_reg, dreg_tag : str):
        if dreg_tag not in self.allocated_registers:
            self.allocated_registers[dreg_tag] = set()
        #print(f"registering existing {dreg_tag} reg: {reg}")
        self.allocated_registers[dreg_tag].add(reg)

    def analyze_preceeding(self, ops: list[lsc_operation]):
        for op in ops:
            for idx in op.reads+op.writes:
                reg = lsc_reg(op.indices[idx].component,
                              op.indices[idx].indices,
                              op.reg_types[idx])
                dreg_tag = determine_dreg_tag(
                        op.tiles[idx].dima, op.tiles[idx].dimb)
                if op.reg_types[idx] in [lrt.address,lrt.offset,lrt.value] and\
                    dreg_tag == 'freg':
                        dreg_tag = 'greg'
                self.register_exists(reg, dreg_tag)

    def register_to_read(self, reg: lsc_reg, dreg_tag : str, pos : int):
        if not reg in self.allocated_registers[dreg_tag]:
            #print(f"registering to alloc {dreg_tag} reg: {reg}")
            if dreg_tag not in self.new_registers:
                self.new_registers[dreg_tag] = set()
            self.new_registers[dreg_tag].add(reg)
        elif reg in self.free_allocated_registers[dreg_tag]:
            #print(f"removing from free {dreg_tag} list: {reg}")
            self.free_allocated_registers[dreg_tag].remove(reg)
        self.rut.add_read(reg, pos)

    def register_to_write(self, reg: lsc_reg, dreg_tag : str, pos : int):
        if not reg in self.allocated_registers[dreg_tag]:
            #print(f"registering to alloc {dreg_tag} reg: {reg}")
            if dreg_tag not in self.new_registers:
                self.new_registers[dreg_tag] = set()
            self.new_registers[dreg_tag].add(reg)
        elif reg in self.free_allocated_registers[dreg_tag]:
            #print(f"removing from free {dreg_tag} list: {reg}")
            self.free_allocated_registers[dreg_tag].remove(reg)
        self.rut.add_write(reg, pos)

    def analyze(self, ops: list[lsc_operation]):
        self.free_allocated_registers = deepcopy(self.allocated_registers)
        self.rut.reset()
        for op_idx,op in enumerate(ops):
            for idx in op.reads:
                reg = lsc_reg(op.indices[idx].component,
                              op.indices[idx].indices,
                              op.reg_types[idx])
                dreg_tag = determine_dreg_tag(
                        op.tiles[idx].dima, op.tiles[idx].dimb)
                if op.reg_types[idx] in [lrt.address,lrt.offset,lrt.value] and\
                    dreg_tag == 'freg':
                        dreg_tag = 'greg'
                self.register_to_read(reg, dreg_tag, op_idx)
            for idx in op.writes:
                reg = lsc_reg(op.indices[idx].component,
                              op.indices[idx].indices,
                              op.reg_types[idx])
                dreg_tag = determine_dreg_tag(
                        op.tiles[idx].dima, op.tiles[idx].dimb)
                if op.reg_types[idx] in [lrt.address,lrt.offset,lrt.value] and\
                    dreg_tag == 'freg':
                        dreg_tag = 'greg'
                self.register_to_write(reg, dreg_tag, op_idx)

    def update_position_map(self, position_map : dict[int,int], src : int, dst : int):

        assert src > dst, "downmove update forbidden"

        #print(f"position map before switching {src} with {dst}:")
        #print("\n".join([f"{k}:{v}" for k,v in position_map.items()]))
        #  d             s 
        # | | | | |...| | |
        #                v
        #  |-------------|
        #  v
        # | | | | |...| | |
        skey = list(position_map.keys())[list(position_map.values()).index(src)]
        # defer updating value

        #  d             s 
        # | | | | |...| | |
        #   \ \ \ \   \ \
        # | | | | |...| | |
        updkeys = [k for k,v in position_map.items() if v >= dst and v < src]
        for k in updkeys:
            position_map[k] += 1


        # update value
        position_map[skey] = dst

        #print(f"position map after switching {src} with {dst}:")
        #print("\n".join([f"{k}:{v}" for k,v in position_map.items()]))


    def move_up(self,
                ops: list[lsc_operation],
                position_map : dict[int,int],
                above_pos : int,
                current_pos : int):

        # first should be furthest up
        cpos = position_map[current_pos]
        apos = position_map[above_pos]

        assert apos < cpos, f"instruction to move up {cpos} is higher than the instruction that defines the ceiling {apos}"

        tmv_pos_list = [current_pos]

        orig_pos = lambda pos : list(position_map.keys())[list(position_map.values()).index(pos)]
        
        pos = cpos-1
        while (pos > apos):
            # We have to look up ops using the non-remapped positions
            first_op = ops[orig_pos(pos)]
            if all([can_reorder(first_op, ops[tmv_k]) for tmv_k in tmv_pos_list]):
                for i,tpos_k in enumerate(tmv_pos_list):
                    tpos = position_map[tpos_k]
                    #print(f"Switching op {pos+i}: {ops[orig_pos(pos+i)]} with op {tpos} : {ops[orig_pos(tpos)]}")
                    self.update_position_map(position_map, tpos, pos+i)
            elif pos-apos > 1:
                #print(f"Can't move {ops[tmv_pos_list[0]]} past {ops[orig_pos(pos)]}")
                key = orig_pos(pos)
                tmv_pos_list = [key]+tmv_pos_list 
            else:
                #print(f"Can't move {ops[tmv_pos_list[0]]} past {ops[orig_pos(pos)]}")
                #print(f"Dependency detected while at the ceiling, breaking")
                break
            pos -=1

    def reorder(self, ops: list[lsc_operation]) -> list[lsc_operation]:
        position_map : dict[int,int] = {i : i for i in range(len(ops))}
        ceiling = 0
        for dreg_tag in ["treg","vreg","freg","greg"]:
            if dreg_tag not in self.new_registers:
                continue

            new_reg_list = list(self.new_registers[dreg_tag])
            minpos_list = [min(set(self.rut.reads.get(reg, [])) | \
                               set(self.rut.writes.get(reg, []))) \
                    for reg in new_reg_list]
            sorted_reg_list = [reg for _,reg in sorted(zip(minpos_list,new_reg_list))]
            for reg in sorted_reg_list:
                pos_set = set(self.rut.reads.get(reg, [])) | \
                          set(self.rut.writes.get(reg, []))
                unsorted_positions = list(pos_set)
                actual_positions = [position_map[k] for k in unsorted_positions]
                sorted_positions = [pos for _,pos in \
                        sorted(zip(actual_positions,unsorted_positions))]
                #actual_positions = [position_map[k] for k in sorted_positions]
                #print(f"Sorted positions: "+ "; ".join([f"{k}:{v}" for k,v in zip(sorted_positions,actual_positions)]))
                
                #Try to move the first one all the way up as well:
                if 0 not in pos_set:
                    sorted_positions = [0]+sorted_positions
                for i in range(len(sorted_positions)-1):
                    cpos = sorted_positions[i+1]
                    if position_map[cpos] <= ceiling:
                        continue
                    apos = sorted_positions[i]
                    # Try moving as close as possible to the previous position
                    #print(f"Moving {cpos}({position_map[cpos]}) as close as possible to {apos}({position_map[apos]})")
                    self.move_up(ops, position_map, apos, cpos)
                ceiling = position_map[sorted_positions[-1]]

        result_ops = len(ops)*[None]
        for k,v in position_map.items():
            result_ops[v] = ops[k]
        return result_ops

    def replace(self, ops: list[lsc_operation], data_registers) -> list[lsc_operation]:


        #print(f"{self.new_registers}")
        #print(f"{self.free_allocated_registers}")

        free_deques = {dreg_tag : deque(free_set) for dreg_tag,free_set \
                in self.free_allocated_registers.items()}



        replace_dict : dict[str,dict[lsc_reg,lsc_reg]] = {
            dreg_tag : dict() for
                dreg_tag in self.free_allocated_registers.keys() }


        hard_used = set()

        result_ops = []
        for op_idx,op in enumerate(ops):
            #print(f"processing OP {op_idx}: {op}")

            regs_this_op = []
            tiles_this_op = []


            for i in op.writes:
                idx = op.indices[i]
                reg = lsc_reg(component=idx.component,
                              indices=idx.indices,
                              rtype=op.reg_types[i])
                if reg in hard_used:
                    continue
                if lrt.data == op.reg_types[i]:
                    dreg_tag = determine_dreg_tag(
                            op.tiles[i].dima, op.tiles[i].dimb)

                    old_reg = reg
                    if old_reg not in replace_dict[dreg_tag]:
                        if free_deques[dreg_tag]:
                            rpl_reg = free_deques[dreg_tag].popleft()
                            assert isinstance(rpl_reg, lsc_reg)
                            replace_dict[dreg_tag][old_reg] = rpl_reg

                            read_del_set = set()
                            for pos in self.rut.reads[old_reg]:
                                if pos > op_idx:
                                    self.rut.add_read(rpl_reg, pos)
                                    read_del_set.add(pos)
                            self.rut.reads[old_reg] = \
                                    self.rut.reads[old_reg].difference(
                                            read_del_set)
                            write_del_set = set()
                            for pos in self.rut.writes[old_reg]:
                                if pos > op_idx:
                                    self.rut.add_write(rpl_reg, pos)
                                    write_del_set.add(pos)
                            self.rut.writes[old_reg] = \
                                    self.rut.writes[old_reg].difference(
                                            write_del_set)


            for i,idx in enumerate(op.indices):
                reg = lsc_reg(component=idx.component,
                              indices=idx.indices,
                              rtype=op.reg_types[i])
                if lrt.data == op.reg_types[i]:
                    dreg_tag = determine_dreg_tag(
                            op.tiles[i].dima, op.tiles[i].dimb)


                    # replace if in map
                    if reg in replace_dict[dreg_tag]:
                        #print(f"replacing: {reg} with {replace_dict[dreg_tag][reg]}")
                        op.indices[i] = replace_dict[dreg_tag][reg].index
                    
                    #print(f"Adding index to used indices this op:{op.indices[i]}")
                    regs_this_op.append(lsc_reg(op.indices[i].component,
                                                op.indices[i].indices,
                                                op.reg_types[i]))
                    tiles_this_op.append(op.tiles[i])

            for used_reg,used_tile in zip(regs_this_op,tiles_this_op):
                dreg_tag = determine_dreg_tag(
                        used_tile.dima,
                        used_tile.dimb)
                if not self.rut.reads[used_reg]:
                    if used_reg not in free_deques[dreg_tag]:
                        #print(f"{used_reg} has no more reads, adding it to free regs")
                        free_deques[dreg_tag].append(used_reg)
                    continue
                if op_idx == self.rut.last_read(used_reg):
                    #print(f"Last read of {used_reg}, releasing it")
                    if used_reg not in free_deques[dreg_tag]:
                        free_deques[dreg_tag].append(used_reg)

            
            hard_used = hard_used.union(set(regs_this_op))

            result_ops.append(deepcopy(op))


        for dreg_tag in self.free_allocated_registers.keys():
            self.free_allocated_registers[dreg_tag] = set(free_deques[dreg_tag])
            for lr in replace_dict[dreg_tag].keys():
                if lr not in self.allocated_registers[dreg_tag]:
                    data_registers[lr.component].remove(lr.indices[0])

        return result_ops
