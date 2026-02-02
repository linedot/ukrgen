from ..stage_param import stage_param
from ..gemm import gemm_context
from ...specializers.asm import lsc_specializer
from .composition import composition_stage

from asmgen.registers import reg_tracker,asm_data_type as adt

class isaparam_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        isa = self.context.params["isa"].value
        isaparams = self.context.gen.get_parameters()

        for name in isaparams:
            pname = f"{isa}-{name}"
            default_value =self.context.gen.get_param_value(name) 
            self.params[pname] = stage_param(
                    value=default_value,
                    default=default_value,
                    description=f"{isa} parameter: {name}"
                    )

    def progress(self):

        isa = self.context.params["isa"].value
        isaparams = self.context.gen.get_parameters()
        for name in isaparams:
            pname = f"{isa}-{name}"
            pval = self.params[pname].value

            self.context.gen.set_parameter(name, pval)


        self.context.rt = reg_tracker([
            ('greg',self.context.gen.max_gregs),
            ('freg',self.context.gen.max_fregs),
            ('vreg',self.context.gen.max_vregs),
            ('treg',self.context.gen.max_tregs(adt.FP64))])

        self.context.specializer = lsc_specializer(
                model=None,
                gen=self.context.gen,
                rt=self.context.rt)

        self.context.params.update(self.params)
        return []
