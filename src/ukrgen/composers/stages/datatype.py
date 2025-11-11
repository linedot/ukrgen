from asmgen.registers import asm_data_type as adt

from .composition import composition_stage

from ..gemm import gemm_context


class datatype_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        component_list = []
        # TODO: There probably should be a different datatype stage for
        #       every kernel
        if "gemm" == context.params["ukr"]:
            # Same component for A and B
            component_list = ["AB","C"]

        self.component_list = component_list

        op = context.params["op"]
        narrow_choices = []
        for sup in context.specializer.op_support_map[op]:
            narrow_choices.append(str(sup.triple.a).replace("asm_data_type.",""))

        wide_choices = []
        for sup in context.specializer.op_support_map[op]:
            wide_choices.append(str(sup.triple.c).replace("asm_data_type.",""))

        component_choices = {
            "AB" : list(set(narrow_choices)),
            "C" : list(set(wide_choices))
        }

        for c in component_list:
            if not component_choices[c]:
                raise ValueError(f"No datatype choices for component {c}")
            name = f"{c}-data-type"
            self.params[name]         = component_choices[c][0]
            self.default_values[name] = component_choices[c][0]
            self.choices[name]        = component_choices[c]


    def progress(self):

        self.context.component_types = {
            k : adt[self.params[f"{k}-data-type"]] for k in self.component_list
        }

        op = self.context.params["op"]
        cts = self.context.component_types

        self.op_support_list = [sup for sup in self.context.specializer.op_support_map[op] if \
                (cts["AB"] == sup.triple.a and \
                 cts["AB"] == sup.triple.b and \
                 cts["C"] == sup.triple.c)
                ]

        if len(self.op_support_list) == 1:
            self.context.sup = self.op_support_list[0]
        elif len(self.op_support_list) == 0:
            raise ValueError(f"No valid op support")

        self.context.op_support_list = self.op_support_list
        self.context.params.update(self.params)
