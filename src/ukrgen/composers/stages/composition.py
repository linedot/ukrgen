from __future__ import annotations

from ..gemm import gemm_context

class composition_stage:

    def __init__(self, context : gemm_context):
        self.context  = context
        self.params : dict[str,str] = dict()
        self.default_values : dict[str,str] = dict()
        self.choices : dict[str,list[str]] = dict()

    def get_default_value(self, name : str):
        return self.default_values[name]

    def get_param_choices(self, name : str):
        return self.choices[name]

    def get_parameter_names(self):
        return list(self.params.keys())

    def set_param(self, name : str, value):
        self.params[name] = value

    def get_param(self, name : str):
        return self.params[name]

    def progress(self) -> list[composition_stage]:
        pass
