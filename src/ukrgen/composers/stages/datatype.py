# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from asmgen.registers import asm_data_type as adt

from .composition import composition_stage
from .variant import variant_stage

from ..gemm import gemm_context
from ..stage_param import stage_param


def extend_adtstr_list(initial_list : list[str]):
    initial_list=list(set(initial_list))

    vs = [adt[c] for c in initial_list]
    for k,v in adt.__members__.items():
        kstripped = k.replace("asm_data_type.","")
        if kstripped not in initial_list and v in vs:
            initial_list.append(kstripped)
    vs = [adt[c].value for c in initial_list]

    return [c for _,c in sorted(zip(vs,initial_list))]

class datatype_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        component_list = []
        # TODO: There probably should be a different datatype stage for
        #       every kernel
        if "gemm" == context.params["ukr"].value:
            # Same component for A and B
            component_list = ["AB","C"]

        self.component_list = component_list

        op = context.params["op"].value
        narrow_choices = []
        for sup in context.specializer.op_support_map[op]:
            narrow_choices.append(str(sup.triple.a).replace("asm_data_type.",""))

        narrow_choices = extend_adtstr_list(narrow_choices)

        wide_choices = []
        for sup in context.specializer.op_support_map[op]:
            wide_choices.append(str(sup.triple.c).replace("asm_data_type.",""))

        wide_choices = extend_adtstr_list(wide_choices)

        component_choices = {
            "AB" : list(set(narrow_choices)),
            "C" : list(set(wide_choices))
        }

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


    def progress(self) -> list[composition_stage]:

        self.context.component_types = {
            k : adt[self.params[f"{k}-data-type"].value] \
                    for k in self.component_list
        }

        op = self.context.params["op"].value
        cts = self.context.component_types

        self.op_support_list = [sup for sup in self.context.specializer.op_support_map[op] if \
                (cts["AB"] == sup.triple.a and \
                 cts["AB"] == sup.triple.b and \
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
