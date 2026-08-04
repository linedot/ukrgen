# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from __future__ import annotations

import re
from typing import Callable,Union

import logging
import traceback
import string

from asmgen.asmblocks.noarch import asmgen
from asmgen.registers import (
    reg_tracker,
    data_reg,
    asm_data_type as adt,
    asm_index_type as ait,
    it_from_dt_samesize,
    adt_is_float,
    adt_is_int,
    adt_triple,
    adt_size,
)
from asmgen.asmblocks.operations import opd3_modifier as mod,widening_method as wm,opd3

from ..components import *
from ..generators import *
from ..models import *
from ..models.offset_mapper import strided_mapper

from ..models.load_store_operations import *
from ..models.loop import lsc_loop

from .special_treg_ldst import special_treg_ldst,lsc_treg_row_store,lsc_treg_row_load

from ..support import op_support


class op_modification:
    """ modification of an lsc_operation
    :param op_match: which operations this modification applies to
    :param component_match: which components this modification applies to
    :param modification: function that accepts the original operation as it's first
                         argument and appends the resulting modified operation(s) to
                         the list that is passed as the second argument. This function
                         should returns True if the modification should be removed
                         from the list or False if the modification should stay
                         in the list
    """
    def __init__(self,
                 op_match : set[type[lsc_operation]],
                 component_match : set[str],
                 reg_type_match : set[lsc_reg_type],
                 modification : Callable[
                     [lsc_operation,
                      list[lsc_operation],
                      list[lsc_operation],
                      list[op_modification]],
                     bool]):
        self.op_match = op_match
        self.component_match = component_match
        self.reg_type_match = reg_type_match
        self.modification = modification

    def __call__(self, op : lsc_operation,
                 prepend_ops : list[lsc_operation],
                 append_ops : list[lsc_operation],
                 new_mods : list[op_modification]) -> bool:

        if not isinstance(op,tuple(self.op_match)):
            return False


        indices = [i for i,idx in enumerate(op.indices)]
        if self.component_match:
            indices = [i for i,idx in enumerate(op.indices) \
                    if idx.component in self.component_match]

        if any(op.reg_types[i] in self.reg_type_match for i in indices):
            return self.modification(op, prepend_ops, append_ops, new_mods)

            
        return False


class lsc_specializer:
    def __init__(self,
                 model : load_store_cpu,
                 gen : asmgen):
        self.model = model
        self.gen = gen

        self.reset_analysis()

        self.op_support_map = {}
        self.get_op_capabilities()

        self.offset_registry = dict()
        self.vindex_registry = dict()


        self.component_triples : set[tuple[str,str,str]] = set()

        # Logging

        self.stridelog = logging.getLogger("STRIDE")

    def set_model(self, model : load_store_cpu):
        self.model = model


    def reset_analysis(self):
        self.regadds = set()
        self.vlenadds = set()
        self.byteadds = set()
        self.ops_used = []
        self.address_registers : dict[str,set[lsc_reg_index]] = dict()
        self.data_registers : dict[str,set[lsc_reg_index]] = dict()
        self.data_tags : dict[str,str] = dict()


    def register_offset(self, component : str, off : lsc_offset):
        if off == lsc_offset.zero_offset():
            # TODO: investigate why this can happen
            return
        #self.stridelog.debug(f"registering offset {off} for component={component}")
        if component not in self.offset_registry:
            self.offset_registry[component] = set()
        if off in self.offset_registry[component]:
            return
        self.offset_registry[component].add(off)

    def register_vindex(self, component : str, stride : lsc_offset):
        if stride == lsc_offset.zero_offset():
            return
        if component not in self.vindex_registry:
            self.vindex_registry[component] = set()
        if stride in self.vindex_registry[component]:
            return
        self.vindex_registry[component].add(stride)


    def analyse(self, ops : list[lsc_operation]):
        for op in ops:
            if isinstance(op, lsc_loop):
                for div in reversed(op.divergences):
                    self.analyse(div.ops)
                self.analyse(op.block)
            if isinstance(op, lsc_addr_add) or \
               isinstance(op, lsc_load) or \
               isinstance(op, lsc_store):
                component = op.indices[0].component
                # reserve registers
                if component not in self.address_registers:
                    self.address_registers[component] = set()

                self.address_registers[component].add(op.addr_idx)

            if isinstance(op, lsc_load) or \
               isinstance(op, lsc_store):
                cr = op.indices[1].component
                # reserve registers
                if cr not in self.data_registers:
                    self.data_registers[cr] = set()

                ca = op.indices[0].component

                dreg_tag = determine_dreg_tag(op.t.dima, op.t.dimb)

                self.data_registers[cr].add(op.res_idx.indices[0])
                self.data_tags[component] = dreg_tag

            if isinstance(op, lsc_addr_add):
                ca = op.indices[0].component
                self.register_offset(component=ca,off=op.off)
            elif isinstance(op, lsc_transformation):
                self.ops_used.append(op.op)

                ctriple = tuple(i.component for i in op.indices)

                self.component_triples.add(ctriple)

                for idx,t in zip(op.indices,op.tiles):

                    c = idx.component
                    residx = idx.indices[0]

                    dreg_tag = determine_dreg_tag(t.dima, t.dimb)

                    if c not in self.data_tags:
                        self.data_tags[c] = dreg_tag
                    else:
                        if self.data_tags[c] != dreg_tag:
                            raise ValueError(f"{op.op} with {residx} is a {dreg_tag}, but is already registered as {self.data_tags[c]}")

                    if c not in self.data_registers:
                        self.data_registers[c] = set()
                    self.data_registers[c].add(residx)


    def pre_specialize(self, ops : list[lsc_operation],
                       component_dts : dict[str,adt]) -> list[lsc_operation]:


        # Determine which indices were used as "c" / accumulators
        acc_components = set()
        mul_components = set()
        for a,b,c in self.component_triples:
            acc_components.add(c)
            mul_components.add(a)
            mul_components.add(b)

        # start with any dt
        widest_dt = next(iter(component_dts.values()))
        narrowest_dt = widest_dt

        for c in acc_components:
            if adt_size(widest_dt) < adt_size(component_dts[c]):
                widest_dt = component_dts[c]
        for c in mul_components:
            if adt_size(narrowest_dt) > adt_size(component_dts[c]):
                narrowest_dt = component_dts[c]

        ways = adt_size(widest_dt)//adt_size(narrowest_dt)
        split_instructions = False
        vec_groups = False
        if ways > 1:
            wms = [getattr(self.gen, op, None).widening_method \
                    for op in self.ops_used if getattr(self.gen, op, None) != None]
            if any([wmtd==wm.SPLIT_INSTRUCTIONS for wmtd in wms]):
                split_instructions = True
            if any([wmtd==wm.VEC_GROUP for wmtd in wms]):
                vec_groups = True



        stls = special_treg_ldst()
        need_stls = False

        modifications=[]

        def widen_data_regs(op : lsc_operation,
                            prepend_ops : list[lsc_operation],
                            append_ops : list[lsc_operation],
                            new_mods : list[op_modification]):
            for i,idx in enumerate(op.indices):
                idx_list = idx.indices
                c = idx.component
                if c in acc_components and lsc_reg_type.data == op.reg_types[i]:
                    op.indices[i].indices[0] *= ways
                    self.data_registers[c].add(op.indices[i].indices[0])
                    for wayreg in range(1,ways):
                        self.data_registers[c].add(op.indices[i].indices[0]+wayreg)
            return False

        def widen_offsets(op : lsc_operation,
                          prepend_ops : list[lsc_operation],
                          append_ops : list[lsc_operation],
                          new_mods : list[op_modification]):
            component = op.addr_idx.component
            if component in acc_components:
                mapper = self.model.offset_mappers[component]
                sizeoff = mapper.get_ldst_size(op.t)
                multoff = op.off.colin(sizeoff)
                addoff = sum([deepcopy(multoff) for j in \
                        range(ways-1)], lsc_offset.zero_offset())

                #print(f"widening {op.off} by adding {addoff}")
                if component in self.offset_registry:
                    if op.off in self.offset_registry[component]:
                        self.offset_registry[component].remove(op.off)
                op.off += addoff
                self.register_offset(component=component,off=op.off)
            return False

        def split_arith_ldst(op : lsc_operation,
                             prepend_ops : list[lsc_operation],
                             append_ops : list[lsc_operation],
                             new_mods : list[op_modification]):
            if need_stls:
                raise NotImplementedError("split instruction and special tile ld/st not implemented")
           
            if isinstance(op, lsc_transformation):
                opstr = op.op
                numbers = re.findall(r'\d+',opstr)
                if numbers:
                    return False

            local_dts = [ component_dts[idx.component] for idx in op.indices]
            min_dtsize = min([adt_size(dt) for dt in local_dts])
            max_dtsize = max([adt_size(dt) for dt in local_dts])

            local_ways = max_dtsize//min_dtsize

            first_op = deepcopy(op)
            for i in range(ways):
                part_op = copy.deepcopy(op)
                for j,idx in enumerate(part_op.indices):
                    idx_list = idx.indices
                    c = idx.component
                    if c in acc_components and lsc_reg_type.data == part_op.reg_types[j]:
                        part_op.indices[j].indices[0] += i
                if isinstance(op, lsc_transformation):
                    if local_ways == ways:
                        part_op.op += f"{i}"
                if isinstance(op, (lsc_load,lsc_store)):
                    baseoff = deepcopy(op.off)
                    ca = op.addr_idx.component
                    sizeoff = self.model.offset_mappers[ca].get_ldst_size(op.t)

                    # This is taken care of by widen_offsets
                    #extent_off = baseoff.colin(sizeoff)
                    #baseoff += sum([extent_off for j in range(ways)],
                    #               lsc_offset.zero_offset())

                    # Only need to add the partial offset
                    addoff = sum([sizeoff for j in range(i)],
                                   lsc_offset.zero_offset())
                    part_op.off = baseoff+addoff

                # TODO: cleaner way to handle this - maybe append_ops should be the
                # only way to return ops?
                if i > 0:
                    append_ops.append(part_op)
                else:
                    first_op = deepcopy(part_op)

            if isinstance(op, lsc_transformation):
                op.op = first_op.op
            if isinstance(op, (lsc_store,lsc_load)):
                op.off = first_op.off
            op.indices = first_op.indices
            return False

        def ensure_ldst_gregstride(op : lsc_operation,
                                   prepend_ops : list[lsc_operation],
                                   append_ops : list[lsc_operation],
                                   new_mods : list[op_modifications]):
            if not isinstance(op, (lsc_load, lsc_store)):
                return False

            ca = op.addr_idx.component
            # Strides
            mapper = self.model.offset_mappers[ca]
            if not isinstance(mapper, strided_mapper):
                return False

            # TODO: arbitrary vectorization direction
            vecdim = mapper.vecdim

            stridx = mapper.stride_indices[vecdim]
            if stridx is None:
                return False

            if ldst_modifier.lane in op.mods:
                return False

            op.stride = lsc_offset({}, [0 for i in range(stridx)]+[1], [], 0)
            self.register_offset(component=ca, off=op.stride)
            can_gregstride = False
            can_gather = False
            can_lane = False
            try:
                self.gen.load_vector_gregstride(
                        areg=self.gen.greg(1),
                        sreg=self.gen.greg(2),
                        vreg=self.gen.vreg(1),
                        dt=adt.FP64)
                can_gregstride = True
            except:
                pass
            try:
                self.gen.load_vector_gather(
                        areg=self.gen.greg(1),
                        vreg=self.gen.vreg(1),
                        offvreg=self.gen.vreg(1),
                        dt=adt.FP64, it=ait.INT64)
                can_gather = True
            except:
                pass
            try:
                self.gen.load_vector_lane(
                        areg=self.gen.greg(1),
                        vreg=self.gen.vreg(1),
                        lane=0,
                        dt=adt.FP64)
                can_lane = True
            except:
                pass

            if can_gregstride:
                return False
            elif not can_gregstride and can_gather:
                self.register_vindex(component=ca, stride=op.stride)
                return False
            elif can_lane:
                # Emulate gregstride with lane loads
                elements = op.t.dima.sd_size*op.t.dimb.sd_size
                for i in range(1,elements):
                    append_ops.append(
                        lsc_addr_add(component=op.addr_idx.component,
                                     addr_idx=op.addr_idx.indices[0],
                                     off=op.stride, 
                                     t=op.t)
                    )
                    lsc_ldst = lsc_load
                    if isinstance(op, lsc_store):
                        lsc_ldst = lsc_store
                    lane_load = lsc_ldst(
                            component=op.addr_idx.component, 
                            res_idx=op.res_idx.indices[0], 
                            addr_idx=op.addr_idx.indices[0], 
                            off=op.off, 
                            stride=op.stride, 
                            t = op.t, 
                            mods=op.mods.union({ldst_modifier.lane}))
                    lane_load.add_property("lane", i)
                    append_ops.append(lane_load)

                op.mods = op.mods.union({ldst_modifier.lane})
                op.add_property("lane", 0)

                addoff = sum([op.stride for i in range(1,elements)],
                             lsc_offset.zero_offset())

                def modify_next_offset(op_mod, prepend_ops, append_ops, new_mods):
                    ca = op_mod.addr_idx.component
                    if op_mod.addr_idx == op.addr_idx:
                        new_off = op_mod.off - addoff
                        if isinstance(op_mod, lsc_addr_add) or \
                                lsc_offset.zero_offset().in_range_of(toff=new_off,
                                offset_range=self.model.ar.offset_ranges[ca][op_mod.addr_idx.indices[0]]):
                            if op_mod.off in self.offset_registry[ca]:
                                self.offset_registry[ca].remove(op_mod.off)
                            op_mod.off = new_off
                            self.register_offset(component=ca,
                                                 off=op_mod.off)
                            return True
                        else:
                            self.register_offset(component=ca, off=addoff)
                            suboff = lsc_offset.zero_offset()-addoff
                            if suboff not in self.offset_registry[ca]:
                                self.register_offset(component=ca, off=suboff)
                            prepend_ops.append(
                                    lsc_addr_add(ca,
                                                 op_mod.addr_idx.indices[0],
                                                 off=suboff,
                                                 t=op.t)
                                    )
                            return True
                    return False
                offsetmod = op_modification(
                    {lsc_addr_add,lsc_load,lsc_store},
                    {op.addr_idx.component},
                    {lsc_reg_type.address},
                    modify_next_offset
                    )
                new_mods.append(offsetmod)
                return False
            else:
                raise RuntimeError("ISA has no gathers, no strided loads and no lane loads. Can't continue")



        modifications.append(op_modification(
            {lsc_load,lsc_store},
            {},
            {lsc_reg_type.address},
            ensure_ldst_gregstride))

        if split_instructions or vec_groups:
            modifications.append(op_modification(
                {lsc_load,lsc_store,lsc_transformation,lsc_zero},
                acc_components,
                {lsc_reg_type.data},
                widen_data_regs))
            modifications.append(op_modification(
                {lsc_load,lsc_store,lsc_addr_add},
                acc_components,
                {lsc_reg_type.address},
                widen_offsets))

        if split_instructions:
            modifications.append(op_modification(
                {lsc_load,lsc_store,lsc_transformation},
                acc_components,
                {lsc_reg_type.data},
                split_arith_ldst))

        result_ops = []
        for op in ops:
            # Apply modifications
            premod_ops = list()
            postmod_ops = list()
            new_mods = list()

            remove_list = [i for i,mod in enumerate(modifications)\
                    if mod(op, premod_ops, postmod_ops, new_mods)]
            for x in reversed(remove_list):
                modifications.pop(x)
            result_ops.extend(premod_ops)
            modifications.extend(deepcopy(new_mods))

            if isinstance(op, lsc_loop):
                for div in reversed(op.divergences):
                    self.pre_specialize(div.ops,triple)
                self.pre_specialize(op.block,triple)

            if need_stls:
                if stls.check_load_flush(op):
                    result_ops.extend(stls.flush_load())
                    print(f"flushing treg loads")
                if stls.check_store_flush(op):
                    result_ops.extend(stls.flush_store())
                    print(f"flushing treg store")

            if isinstance(op, (lsc_load, lsc_store)):
                ca = op.addr_idx.component
                # Strides
                mapper = self.model.offset_mappers[ca]
                
                # TODO: arbitrary vectorization direction
                vecdim = 0

                for a,b,c in self.component_triples:
                    if b == ca:
                        vecdim = 1

                if op.t.is_tile:
                    action = "load"
                    if isinstance(op, lsc_store):
                        action = "store"
                    try:
                        lsfunc = getattr(self.gen,f"{action}_tile")
                        lsfunc(areg=self.gen.greg(0),
                               treg=self.gen.treg(0),
                               dt=component_dts[op.addr_idx.component])
                        result_ops.append(op)
                    except:
                        print(f"adding treg {action}")
                        need_stls = True
                        addfunc = getattr(stls, f"add_treg_{action}")
                        addfunc(op, triple[op.addr_idx.component])
                    continue

            has_c = False
            for idx in op.indices:
                c = idx.component
                if c in acc_components:
                    has_c = True

            if has_c and vec_groups and isinstance(op, (lsc_load,lsc_store,lsc_zero)):

                for i in range(ways):
                    group_op = copy.deepcopy(op)
                    for j,idx in enumerate(group_op.indices):
                        c = idx.component
                        idx_list = idx.indices
                        if c in acc_components and \
                                lsc_reg_type.data == group_op.reg_types[j]:
                            group_op.indices[j].indices[0] += i

                    if isinstance(op, (lsc_load,lsc_store)):
                        ca = op.addr_idx.component
                        baseoff = deepcopy(group_op.off)
                        sizeoff = self.model.offset_mappers[ca].get_ldst_size(op.t)
                        extent_off = baseoff.colin(sizeoff)
                        baseoff += sum([extent_off for j in \
                                range(ways)],lsc_offset.zero_offset())

                        # TODO: mechanism for sending this back to addr_resolver.
                        #       For now assume this works
                        addoff = sum([sizeoff for j in \
                                range(i)],lsc_offset.zero_offset())

                        gca = group_op.addr_idx.component
                        offrange = self.model.ar.offset_ranges[gca][group_op.addr_idx.indices[0]]
                        allowed = baseoff.in_range_of(
                            toff=baseoff+addoff,
                            offset_range=offrange)
                        if not allowed:
                            result_ops.append(
                                    lsc_addr_add(
                                        ca,
                                        op.addr_idx.indices[0],
                                        addoff,
                                        group_op.t))
                            group_op.off = baseoff

                            mod_idx = op.addr_idx
                            def subtract_addoff(op, prepend_ops,
                                                append_ops, new_mods):
                                ca = op.addr_idx.component
                                if op.addr_idx == mod_idx:
                                    #print(f"subtracting {addoff} from {op.off}")
                                    if op.off in self.offset_registry[ca]:
                                        self.offset_registry[ca].remove(op.off)
                                    op.off -= addoff
                                    self.register_offset(component=ca,
                                                         off=op.off)
                                    return True
                                return False

                            modifications.append(
                                    op_modification(
                                        op_match={lsc_addr_add,lsc_load,lsc_store},
                                        component_match=acc_components, 
                                        reg_type_match={lsc_reg_type.address},
                                        modification=subtract_addoff)
                                    )
                        else:
                            group_op.off = baseoff+addoff

                    result_ops.append(group_op)
            else:
                result_ops.append(op)

            result_ops.extend(postmod_ops)
        if need_stls:
            result_ops.extend(stls.flush_load())
            result_ops.extend(stls.flush_store())
        return result_ops
