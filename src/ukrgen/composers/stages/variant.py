from .composition import composition_stage
from ..gemm import gemm_context

from ...specializers.asm import op_support

class variant_stage(composition_stage):
    def __init__(self, context : gemm_context,
                 op_support_list : list[op_support]):
        super().__init__(context)

        self.op_support_list = op_support_list


        self.params["variant"] = 0
        self.default_values["variant"] = 0
        self.choices["variant"] = list(range(len(op_support_list)))

    def progress(self):
        self.context.sup = self.op_support_list[self.params["variant"]]

        self.context.params.update(self.params)
