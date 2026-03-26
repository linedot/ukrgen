# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from asmgen.registers import asm_data_type as adt

from .stage import stage
from .variant import variant_stage

from ..ukr_context import ukr_context
from ..stage_param import stage_param

from .ukr import ukr_composition_map

import itertools


def extend_adtstr_list(initial_list : list[str]):
    initial_list=list(set(initial_list))

    initial_dts = {adt[name] for name in initial_list}
    initial_values = {dt.value for dt in initial_dts}

    extended_dts = [dt for dt in adt if dt.value in initial_values]
    extended_dts.sort(key= lambda dt: (dt.value, dt.name))

    return [dt.name for dt in extended_dts]


class datatype_stage(stage):
    def __init__(self, context : ukr_context):
        super().__init__(context)



        ukr = context.params["ukr"].value
        composition = ukr_composition_map[ukr]

        component_list = []

        component_list = composition.get_parameterized_components()
        self.component_list = component_list

        # Get ISA supported datatypes
        op = context.params["op"].value

        a_choices = []
        for sup in context.specializer.op_support_map[op]:
            a_choices.append(sup.triple.a.name)
        a_choices = extend_adtstr_list(a_choices)

        b_choices = []
        for sup in context.specializer.op_support_map[op]:
            b_choices.append(sup.triple.b.name)
        b_choices = extend_adtstr_list(b_choices)

        c_choices = []
        for sup in context.specializer.op_support_map[op]:
            c_choices.append(sup.triple.c.name)
        c_choices = extend_adtstr_list(c_choices)


        # Set choices per component
        component_choices = {}
        for c in component_list:
            choices = set()
            sup_tiles = composition.get_component_sup_tiles(c)
            if "a" in sup_tiles:
                choices.update(a_choices)
            if "b" in sup_tiles:
                choices.update(b_choices)
            if "c" in sup_tiles:
                choices.update(c_choices)

            component_choices[c] = list(choices)

        for c in component_list:
            if not component_choices[c]:
                raise ValueError(f"No datatype choices for component {c}")
            name = f"{c}-data-type"

            self.params[name] = stage_param(
                    value=None,
                    description=f"Data type for component {c}",
                    choices=component_choices[c],
                    required=True
                    )


    def progress(self) -> list[stage]:

        self.context.component_types = {
            k : adt[self.params[f"{k}-data-type"].value] \
                    for k in self.component_list
        }

        op = self.context.params["op"].value
        cdts = self.context.component_types
        ukr = self.context.params["ukr"].value

        composition = ukr_composition_map[ukr]
        components = composition.get_components()
        sups = self.context.specializer.op_support_map[op]


        sup_tile_components = {
            "a" : set(),
            "b" : set(),
            "c" : set(),
        }

        for c in components:
            real_c = composition.get_component_reference(c)
            sup_tiles = composition.get_component_sup_tiles(real_c)
            for stile in sup_tiles:
                if stile in sup_tile_components:
                    sup_tile_components[stile].add(c)
            
            # also squeeze in this assignment into this loop
            cdts[c] = cdts[real_c]

        # If a sup tile is unmapped, this operand will be ignored
        # in that case we'll still add the sup
        for sup_tile in sup_tile_components:
            if not sup_tile_components[sup_tile]:
                sup_tile_components[sup_tile].add(None)
        
        self.op_support_list = []
        for sup in sups:
            # example:
            # component : tileset
            # A : {a,b}
            # B : {a,b}
            # C : {c}
            # AB : {c}
            # alpha->C : {c}
            # beta->C : {c}

            # combinations:
            # A, B, C
            # B, A, C
            # A, B, AB
            # B, A, AB
            # A, B, alpha
            # B, A, alpha
            # A, B, beta
            # B, A, beta

            # sup = fp64, fp64, fp64
            # all cdts fp64:
            #
            # ==> add

            # sup = fp16, fp16, fp32
            # cdts: A,B: FP16, rest: FP32
            #
            # ==> add
            # sup = fp32, fp32, fp32
            # cdts: A,B: FP16, rest: FP32
            #
            # ==> don't add

            # sup = fp8e5m2, fp16, fp32
            # cdts: A: fp16, B: fp8e5m2, C: fp32
            # combinations:
            # A, B, C     <- invalid
            # B, A, C     <- valid
            # A, B, AB    <- invalid
            # B, A, AB    <- valid
            # A, B, alpha <- invalid
            # B, A, alpha <- valid
            # A, B, beta  <- invalid
            # B, A, beta  <- valid
            #
            # ==> add
            
            components_mapped = set()
            for a,b,c in itertools.product(sup_tile_components['a'],
                                           sup_tile_components['b'],
                                           sup_tile_components['c']):
                combination_components = [x for x in (a,b,c) if x is not None]

                if len({a,b,c}) < len(combination_components):
                    continue

                if ((sup.triple.a == cdts[a]) and
                    (sup.triple.b == cdts[b]) and
                    (sup.triple.c == cdts[c])):
                    components_mapped.update(combination_components)

            if components_mapped == set(components):
                self.op_support_list.append(sup)

        self.context.component_dts = cdts

        if len(self.op_support_list) == 1:
            self.context.sup = self.op_support_list[0]
        elif len(self.op_support_list) == 0:
            raise ValueError(f"No valid op support")

        self.context.op_support_list = self.op_support_list
        self.context.params.update(self.params)

        if len(self.op_support_list) > 1:
            return [variant_stage]
        else:
            return list()
