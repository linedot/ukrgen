# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from enum import Enum,auto


from ..models.load_store_cpu import (
        tile
        )
from ..models.load_store_operations import (
        lsc_reg_type,
        lsc_operation,
     )

from ..models.lsc.index import lsc_reg_index

class reg_compare:
    def __init__(self, ttype : lsc_reg_type,
                 index : lsc_reg_index,
                 t : tile = None):

        assert isinstance(index, lsc_reg_index)
        assert isinstance(ttype, lsc_reg_type)
        self.ttype = ttype
        self.index = index
        self.t = t

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



class simple_dependency_scheduler:
    def __init__(self,
                 rar : int = 0,
                 raw : int = 10,
                 war : int = 0,
                 waw : int = 10,
                 patterns : list[str] = [],
                 reg_types : set[lsc_reg_type] = {lsc_reg_type.data},
                 debug_on : bool = False
                 ):
        self.rar = rar
        self.raw = raw
        self.war = war
        self.waw = waw
        self.patterns = patterns
        self.reg_types = reg_types
        self.debug_on = debug_on

    def debug(self, msg : str):
        if self.debug_on:
            print(msg)

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
                        t=next_op.tiles[i]) for i in next_op.reads \
                                if next_op.reg_types[i] in self.reg_types}
        next_op_writes = {
            reg_compare(ttype=next_op.reg_types[i],
                        index=next_op.indices[i],
                        t=next_op.tiles[i]) for i in next_op.writes \
                                if next_op.reg_types[i] in self.reg_types}

        cur_op_reads = {
            reg_compare(ttype=cur_op.reg_types[i],
                        index=cur_op.indices[i],
                        t=cur_op.tiles[i]) for i in cur_op.reads \
                                if cur_op.reg_types[i] in self.reg_types}
        cur_op_writes = {
            reg_compare(ttype=cur_op.reg_types[i],
                        index=cur_op.indices[i],
                        t=cur_op.tiles[i]) for i in cur_op.writes \
                                if cur_op.reg_types[i] in self.reg_types}

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
            self.debug(f"rar={self.rar-distance}(distance={distance}) between {cur_op} and {next_op}")
            depends = True
        if checks[1] and len(raws)>0:
            move_up = max(move_up,max(0,self.raw - distance))
            self.debug(f"raw={self.raw-distance}(distance={distance}) between {cur_op} and {next_op}")
            depends = True
        if checks[2] and len(wars)>0:
            move_up = max(move_up,max(0,self.war - distance))
            self.debug(f"war={self.war-distance}(distance={distance}) between {cur_op} and {next_op}")
            depends = True
        if checks[3] and len(waws)>0:
            move_up = max(move_up,max(0,self.waw - distance))
            self.debug(f"waw={self.waw-distance}(distance={distance}) between {cur_op} and {next_op}")
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
                            self.debug(f"{scheduled[test_m_idx][1]} depends on {prev_op} (S-check)")
                        else:
                            self.debug(f"{scheduled[test_m_idx][1]} does not depend on {prev_op} (S-check)")
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
