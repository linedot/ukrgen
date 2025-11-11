from .composition import composition_stage
from ..gemm import gemm_context

from ...specializers.asm import op_support

class dimension_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        for dim in ["m","n","k"]:
            self.params[dim] = None
            self.default_values[dim] = None
            self.choices[dim] = None

        self.params["vecdir"] = "M"
        self.default_values["vecdir"] = "M"
        self.choices["vecdir"] = ["M","N"] # TODO: K

        self.params["order"] = "mnkMNK"
        self.default_values["order"] = "mnkMNK"
        self.choices["order"] = None

    def progress(self):
        # Do we need to unvec?
        if self.context.params["op"] == "fma" and \
                self.context.sup.b_tile.is_vector and \
                self.context.sup.a_tile.is_vector:

            self.context.needs_unvec = True

        self.context.params["ma"] = self.params["m"]
        self.context.params["mc"] = self.params["m"]
        self.context.params["nb"] = self.params["n"]
        self.context.params["nc"] = self.params["n"]


        self.context.params.update(self.params)
