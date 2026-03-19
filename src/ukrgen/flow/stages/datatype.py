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
        cts = self.context.component_types
        ukr = self.context.params["ukr"].value

        #TODO: This needs some kind of generalized logic
        if ukr in {"gemm","mm"}:
            self.op_support_list = [sup for sup in\
                    self.context.specializer.op_support_map[op] if \
                    (cts["A"] == sup.triple.a and \
                     cts["B"] == sup.triple.b and \
                     cts["C"] == sup.triple.c)
                    ]

            self.context.component_dts = {
                "A" : adt[self.params["AB-data-type"].value],
                "B" : adt[self.params["AB-data-type"].value],
                "AB" : adt[self.params["C-data-type"].value],
                "C" : adt[self.params["C-data-type"].value],
                "alpha" : adt[self.params["C-data-type"].value],
                "beta" : adt[self.params["C-data-type"].value]
            }
        if ukr in {"pack"}:
            self.op_support_list = [sup for sup in\
                    self.context.specializer.op_support_map[op] if \
                    (cts["X"] == sup.triple.a and \
                     cts["X"] == sup.triple.b and \
                     cts["Y"] == sup.triple.c)
                    ]

            self.context.component_dts = {
                "X" : adt[self.params["X-data-type"].value],
                "Y" : adt[self.params["Y-data-type"].value],
                "kappa" : adt[self.params["Y-data-type"].value]
            }

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
