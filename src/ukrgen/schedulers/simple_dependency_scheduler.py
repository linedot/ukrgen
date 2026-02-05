# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from __future__ import annotations

from ..models.load_store_cpu import (
        tile
        )
from ..models.load_store_operations import (
        lsc_reg_type,
        lsc_operation,
     )

from ..models.lsc.index import lsc_reg_index

from ..components.tile import tile
from ..components.tile import determine_dreg_tag

from asmgen.registers import greg_base,data_reg

from typing import Type
from dataclasses import dataclass
import logging
from enum import Enum,auto

class reg_compare:
    def __init__(self, ttype : lsc_reg_type,
                 index : lsc_reg_index,
                 t : tile = None):

        assert isinstance(index, lsc_reg_index)
        assert isinstance(ttype, lsc_reg_type)
        self.ttype = ttype
        self.index = index
        self.t : tile = t

    def __eq__(self, other):
        if len(self.index.indices) != len(other.index.indices):
            return False
        if self.ttype != other.ttype:
            return False
        if self.index.component != other.index.component:
            return False
        if any([idx1 != idx2 for idx1,idx2 \
                in zip(self.index.indices,other.index.indices)]):
            return False

        return True

    def __hash__(self):
        h =  hash((self.ttype,self.index))
        #print(f"hash of {self.__str__()} : {h}")
        return h
    def __str__(self):
        result = "RES:"
        if self.ttype == lsc_reg_type.address:
            result = "ADDR:"
        result += str(self.index)
        return result
    def __repr__(self):
        return self.__str__()


class dependency_type(Enum):
    RAR = auto()
    RAW = auto()
    WAR = auto()
    WAW = auto()

def get_dependencies(op1 : lsc_operation, op2: lsc_operation) ->\
        dict[dependency_type,set[reg_compare]]:


    op2_reads = {
        reg_compare(ttype=op2.reg_types[i],
                    index=op2.indices[i],
                    t=op2.tiles[i]) for i in op2.reads}
    op2_writes = {
        reg_compare(ttype=op2.reg_types[i],
                    index=op2.indices[i],
                    t=op2.tiles[i]) for i in op2.writes}

    op1_reads = {
        reg_compare(ttype=op1.reg_types[i],
                    index=op1.indices[i],
                    t=op1.tiles[i]) for i in op1.reads}
    op1_writes = {
        reg_compare(ttype=op1.reg_types[i],
                    index=op1.indices[i],
                    t=op1.tiles[i]) for i in op1.writes}


    return {
       dependency_type.RAR : set(op2_reads) & set(op1_reads),
       dependency_type.RAW : set(op2_reads) & set(op1_writes),
       dependency_type.WAR : set(op2_writes) & set(op1_reads),
       dependency_type.WAW : set(op2_writes) & set(op1_writes),
    }

@dataclass
class distance_specification:
    """
    Specifies the distance for the scheduler to maintain between instructions
    and the circumstances under which to maintain it

    :param dep_type: Type of dependency (read-after-read, read-after-write,
                     etc..) for which the distance is to be maintained. When
                     set to None, applies to all dependency types
    :type dep_type: class:`dependency_type|None`
    :param reg_type: type of register in the load-store cpu model
                     (data, address, value, ...). When set to None,
                     applies to all register types
    :type reg_type: class:`lsc_reg_type|None`
    :param asm_reg_tag: type of asm reg type as string, can be "greg", "freg",
                        "vreg" or "treg". When set to None, applies to all
                        asm register types.
                        Note: to apply to all data reg types ("freg","vreg",
                        "treg"), set this to None and set reg_type to data
    :type asm_reg_tag: str|None
    :param op1_type: first operation in the dependency chain link. When set to
                     None, applies to all operations
    :type op1_type: class:`Type[lsc_operation]|None`
    :param op2_type: second operation in the dependency chain link. When set
                     to None, applies to all operations
    :type op2_type: class:`Type[lsc_operation]|None`
    :param tr1_name: if the first operation is an lsc_transformation, this 
                     string specifies the exact transformation, i.e fma, dota,
                     fmul, etc... When set to None, applies to all 
                     transformations
    :type tr1_name: str|None
    :param tr2_name: if the first operation is an lsc_transformation, this 
                     string specifies the exact transformation, i.e fma, dota,
                     fmul, etc... When set to None, applies to all 
                     transformations
    :type tr2_name: str|None
    :param distance: distance, in independent instructions, to maintain.
    :type distance: int
    """
    dep_type     : dependency_type|None
    reg_type     : lsc_reg_type|None
    asm_reg_tag  : str|None
    op1_type     : Type[lsc_operation]|None
    op2_type     : Type[lsc_operation]|None
    tr1_name     : str|None
    tr2_name     : str|None
    distance     : int

    @classmethod
    def choose_from_enum(cls, choice : str, enum : Type[Enum]) -> Enum:
        result = None
        valid_values = [str(e).split('.')[-1] for e in enum]
        if not choice:
            pass
        elif choice not in valid_values:
            choices_str = ",".join([lower(d) for d in valid_values])
            raise ValueError(
                (f"Invalid spec: {choice}. Must be one of"
                 f" [{choices_str}]"
                 ))
        else:
            result = enum[choice]

        return result

    @classmethod
    def from_string(cls, specstr : str) -> distance_specification:

        split_spec = specstr.split(":")


        dep_type = cls.choose_from_enum(choice=split_spec[0].upper(),
                                        enum=dependency_type)
        reg_type = cls.choose_from_enum(choice=split_spec[1],
                                        enum=lsc_reg_type)

        asm_reg_tag = None

        tag_str = split_spec[2]
        valid_tags = ["greg","freg","vreg","treg"]
        if asm_reg_tag:
            if tag_str not in valid_tags:
                choices_str = ",".join(valid_tags)
                raise ValueError(f"tag {tag_str} not in [{choices_str}]")
            asm_reg_tag = tag_str

        op1_str = split_spec[3]
        op2_str = split_spec[4]
        op_types = {
            "trf" : lsc_transformation,
            "fma" : lsc_transformation,
            "fmul" : lsc_transformation,
            "dota" : lsc_transformation,
            "opa" : lsc_transformation,
            "mma" : lsc_transformation,
            "ld" : lsc_load,
            "st" : lsc_store,
            "z" : lsc_zero,
            "aadd" : lsc_addr_add,
            "oadd" : lsc_add_val_off,
        }

        op1_type = None
        tr1_name = None
        op2_type = None
        tr2_name = None

        if op1_str:
            if op1_str not in op_types.keys():
                raise ValueError(f"Unknown op \"{op1_type_str}\"")
            op1_type = op_types[op1_str]
        if op2_str:
            if op2_str not in op_types.keys():
                raise ValueError(f"Unknown op \"{op2_type_str}\"")
            op2_type = op_types[op2_str]

        if op1_type == lsc_transformation:
            tr1_name = op1_str if "trf" != op1_str else None
        if op2_type == lsc_transformation:
            tr2_name = op2_str if "trf" != op2_str else None

        distance = int(split_spec[5])

        return  distance_specification(
                dep_type=dep_type, 
                reg_type=reg_type, 
                asm_reg_tag=asm_reg_tag,
                op1_type=op1_type, 
                op2_type=op2_type, 
                tr1_name=tr1_name, 
                tr2_name=tr2_name, 
                distance=distance)

    def apply(self,
              op1 : lsc_operation,
              op2 : lsc_operation) -> int:
        """
        Check if the distance constraint applies between the specified ops
        and if it does, returns the distance, otherwise returns 0
        """

        # Types

        if self.op1_type is not None and \
          not isinstance(op1, self.op1_type):
            return 0

        if self.op2_type is not None and \
          not isinstance(op2, self.op2_type):
            return 0
        
        # Exact transformations

        if self.op1_type == lsc_transformation:
            if self.tr1_name is not None:
                if self.tr1_name != op1.op:
                    return 0

        if self.op2_type == lsc_transformation:
            if self.tr2_name is not None:
                if self.tr2_name != op2.op:
                    return 0

        deps = get_dependencies(op1=op1, op2=op2)

        result = 0

        # registers
        for dep,regs in deps.items():
            if not regs:
                continue
            if self.dep_type is not None and self.dep_type != dep:
                continue
            for rc in regs:
                if self.reg_type is not None and self.reg_type != rc.ttype:
                    continue
                
                asm_reg_tag = determine_dreg_tag(dima=rc.t.dima, dimb=rc.t.dimb)
                if self.asm_reg_tag is not None and self.asm_reg_tag != asm_reg_tag:
                    continue
                    
                result = self.distance


        return result


class simple_dependency_scheduler:
    def __init__(self,
                 rar : int = 0,
                 raw : int = 10,
                 war : int = 0,
                 waw : int = 10,
                 patterns : list[str] = [],
                 reg_types : set[lsc_reg_type] = {lsc_reg_type.data},
                 ):
        self.rar = rar
        self.raw = raw
        self.war = war
        self.waw = waw
        self.patterns = patterns
        self.reg_types = reg_types

        self.debug = logging.getLogger("SCHED").debug

    def get_move_up(self,
                    next_op : lsc_operation,
                    cur_op : lsc_operation,
                    distance : int,
                    checks : tuple[bool,bool,bool,bool] = (True,True,True,True) ) -> int:
        move_up = 0
        depends = False

        next_op_reads = {
            reg_compare(ttype=next_op.reg_types[i],
                        index=next_op.indices[i],
                        t=next_op.tiles[i]) for i in next_op.reads}
        next_op_writes = {
            reg_compare(ttype=next_op.reg_types[i],
                        index=next_op.indices[i],
                        t=next_op.tiles[i]) for i in next_op.writes}

        cur_op_reads = {
            reg_compare(ttype=cur_op.reg_types[i],
                        index=cur_op.indices[i],
                        t=cur_op.tiles[i]) for i in cur_op.reads}
        cur_op_writes = {
            reg_compare(ttype=cur_op.reg_types[i],
                        index=cur_op.indices[i],
                        t=cur_op.tiles[i]) for i in cur_op.writes}

        #self.debug(f"cur_op = {cur_op}. reads: {cur_op_reads}")
        #self.debug(f"cur_op = {cur_op}. writes: {cur_op_writes}")
        #self.debug(f"next_op = {next_op}. reads: {next_op_reads}")
        #self.debug(f"next_op = {next_op}. writes: {next_op_writes}")

        #self.debug(f"{checks}")

        rars = set(next_op_reads) & set(cur_op_reads)
        raws = set(next_op_reads) & set(cur_op_writes)
        wars = set(next_op_writes) & set(cur_op_reads)
        waws = set(next_op_writes) & set(cur_op_writes)

        #self.debug(f"RAR: {rars}")
        #self.debug(f"RAW: {raws}")
        #self.debug(f"WAR: {wars}")
        #self.debug(f"WAW: {waws}")

        if checks[0] and len(rars)>0:
            move_up = max(move_up,max(0,self.rar - distance))
            if not any([dep.ttype in self.reg_types for dep in rars]):
                move_up = 0
            self.debug((f"rar={self.rar-distance}(distance={distance}) between \n"
                        f"  {cur_op}\n"
                         "   and\n"
                        f"  {next_op}"))
            #depends = True
        if checks[1] and len(raws)>0:
            move_up = max(move_up,max(0,self.raw - distance))
            if not any([dep.ttype in self.reg_types for dep in raws]):
                move_up = 0
            self.debug((f"raw={self.raw-distance}(distance={distance}) between \n"
                        f"  {cur_op}\n"
                         "   and\n"
                        f"  {next_op}"))
            depends = True
        if checks[2] and len(wars)>0:
            move_up = max(move_up,max(0,self.war - distance))
            if not any([dep.ttype in self.reg_types for dep in wars]):
                move_up = 0
            self.debug((f"war={self.war-distance}(distance={distance}) between \n"
                        f"  {cur_op}\n"
                         "   and\n"
                        f"  {next_op}"))
            depends = True
        if checks[3] and len(waws)>0:
            move_up = max(move_up,max(0,self.waw - distance))
            if not any([dep.ttype in self.reg_types for dep in waws]):
                move_up = 0
            self.debug((f"waw={self.waw-distance}(distance={distance}) between \n"
                        f"  {cur_op}\n"
                         "   and\n"
                        f"  {next_op}"))
            depends = True
        return depends,move_up


    def depschedule(self, ops: list[lsc_operation], loop : bool) -> list[lsc_operation]:
        scheduled = []
        lookforward = 0
        if loop:
            lookforward = max(self.rar,self.raw,self.war,self.waw)

        opslf = ops+ops[:lookforward]
        for cur_idx in range(len(ops)+lookforward):
            # update the lookforward when at end of loop
            if loop and cur_idx == len(ops)-1:
                opslf[len(ops):] = [sop[1] for sop in scheduled[:lookforward]]
            cur_op = opslf[cur_idx]
            move_indices = []
            # Check for distances to dependencies and add to move list
            # along with the distance to move
            for sched_idx,(orig_idx,sched_op) in enumerate(scheduled):
                distance = len(scheduled)-1-sched_idx
                depends,move_up = self.get_move_up(next_op=cur_op,
                                           cur_op=sched_op,
                                           distance=distance)
                if depends and distance == 0:
                    move_up = 1
                if move_up > 0:
                    # save with original index, which we use to filter the lookforward
                    # in the end
                    move_indices.append((sched_idx,move_up))

            if not move_indices:
                scheduled.append((cur_idx,cur_op))
                continue

            self.debug("Currently scheduled: ")
            self.debug("  "+"\n  ".join([f"pos {scheduled[idx][0]}: "+\
                    str(scheduled[idx][1]) for idx in range(len(scheduled))]))
            self.debug(f"instructions that need to be moved up :")
            for idx,places in move_indices:
                sched_pos,sched_op = scheduled[idx]
                self.debug(f"  pos {sched_pos} up {places} places : {sched_op}")
                self.debug(f"    (reads {sched_op.reads} and writes {sched_op.writes})")
            self.debug(f"because the next instruction is:")
            self.debug(f"  {cur_op}")
            self.debug(f"    (reads {cur_op.reads} and writes {cur_op.writes})")


            midx = 0
            while midx < len(move_indices):
                self.debug(f"instructions that need to be moved up :")
                for idx,places in move_indices:
                    sched_pos,sched_op = scheduled[idx]
                    self.debug(f"  pos {sched_pos} up {places} places : {sched_op}")
                    self.debug(f"    (reads {sched_op.reads} and writes {sched_op.writes})")
                idx,move_up = move_indices[midx]
                if move_up == 0:
                    midx += 1
                    continue
                insts_that_move = sum([1 if mup > 0 else 0 for sidx,mup in move_indices])
                resched_idx,resched_op = scheduled[idx]
                midx_add = 1
                for move_idx in range(move_up):
                    if idx == 0:
                        # TODO: allow suboptimal solution?
                        raise RuntimeError("Instruction at top of schedule and min distances not satisfied")
                    prev_idx,prev_op = scheduled[idx-1]


                    # if prev_op is also in move_indices:
                    #  - it has a dependency with same op as this op
                    #  - it was moved to just the minimum distance
                    #  - so, if we swap with it, it will be below min distance
                    #  - therefore increase it's move_up by move_up-move_idx,
                    #    decrement midx and break out to outer loop
                    if midx > 0:
                        midx_found = None
                        for midx_check,(test_m_idx,_) in enumerate(move_indices):
                            if scheduled[test_m_idx][0] == prev_idx:
                                midx_found = midx_check
                        if None != midx_found:
                            assert midx > midx_found
                            prev_m_idx,prev_m_move_up = move_indices[midx_found]
                            move_indices[midx_found] = (prev_m_idx, prev_m_move_up + (move_up-move_idx))
                            midx_add = midx_found-midx
                            break

                    max_preceding_to_test = min(cur_idx,
                                                max(self.rar,self.raw,self.war,self.waw),
                                                len(scheduled))

                    # check if the previous instruction would depend on a scheduled instruction
                    # if swapped
                    prev_breakout = False
                    add_distance=0
                    for prev_cur_idx,prev_cur_op in \
                            enumerate(opslf[cur_idx-max_preceding_to_test:cur_idx+1]):
                        if prev_cur_op == prev_op:
                            break
                        # test prev_op for dependency on cur_op with distance-1
                        # if that results in a moveup of >0, prepend it to move_indices
                        # decrement midx and break out
                        depends,move_up_prev = self.get_move_up(
                                next_op=prev_cur_op,
                                cur_op=prev_op,
                                distance=len(move_indices)-insts_that_move-add_distance+prev_cur_idx)
                        if move_up_prev > 0:
                            self.debug(f"{prev_cur_op} depends on {prev_op}")
                            move_indices.insert(0,(idx-1,move_up_prev))
                            midx_add = 0
                            prev_breakout = True
                            break
                    if prev_breakout:
                        break
                        
                        

                    depends = False
                    move_up_prev_max = 0
                    # test against all to be rescheduled from this one
                    for midx_check,(test_m_idx,move_up_left) in enumerate(move_indices[midx:]):
                        depends_this,move_up_prev = \
                                self.get_move_up(
                                        next_op=scheduled[test_m_idx][1],
                                        cur_op=prev_op,
                                        distance=midx_check,
                                        # read-after-read is irrelevant
                                        checks=[False,True,True,True])
                        if depends_this:
                            self.debug((f">FOUND< Dependency of\n"
                                        f"  {scheduled[test_m_idx][1]}\n"
                                        f"    on\n"
                                        f"  {prev_op} (S-check)"))
                        else:
                            self.debug((f">NO< dependency of\n"
                                        f"  {scheduled[test_m_idx][1]}\n"
                                        f"    on\n"
                                        f"  {prev_op} (S-check)"))
                        # op is dependent if it depends on ANY ops from move_indices
                        depends = any([depends,depends_this])
                        move_up_prev_max = max(move_up_prev_max,move_up_prev)

                        
                    
                    # We can't swap the instructions, so we need to move
                    # the previous instruction up as well
                    if depends:
                        self.debug(f"inserting {prev_op} into move list at position {midx}")
                        # TODO: better data structure than list
                        move_indices.insert(midx,(idx-1, max(move_up-move_idx,move_up_prev_max) ))
                        # Don't increment the index (it points to the prev op now)
                        midx_add = 0
                        # And stop processing this op
                        break

                    self.debug(f"swapped {prev_op} and {resched_op}")
                    scheduled[idx] = (prev_idx,prev_op)
                    scheduled[idx-1] = (resched_idx,resched_op)
                    idx -= 1
                    move_indices[midx] = (idx,move_up-move_idx-1)
                    # We might need a check for rar here after the swap??
                midx += midx_add

            self.debug("After rescheduling: ")
            self.debug("  "+"\n  ".join([f"pos {scheduled[idx][0]}: "+\
                    str(scheduled[idx][1]) for idx in range(len(scheduled))]))


            self.debug("============================================================")
            scheduled.append((cur_idx,cur_op))

        final_schedule = [sop for sop in scheduled if sop[0] < len(ops)]

        self.debug("Final Schedule (without lookforward): ")
        self.debug("  "+"\n  ".join([f"pos {final_schedule[idx][0]}: "+\
                str(final_schedule[idx][1]) for idx in range(len(final_schedule))]))

        return [sop[1] for sop in final_schedule]

        

    def __call__(self, ops : list[lsc_operation], loop : bool = True):
        reordered_ops = []
        queue = []
        
        return self.depschedule(ops=ops, loop=loop)
