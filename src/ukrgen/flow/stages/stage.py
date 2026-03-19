# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from __future__ import annotations

from ..ukr_context import ukr_context

from ..stage_param import stage_param

class stage:

    def __init__(self, context : ukr_context):
        self.context  = context
        self.params : dict[str,stage_param] = dict()

    def get_parameter_names(self):
        return list(self.params.keys())

    def set_param(self, name : str, value):
        self.params[name].value = value

    def get_param(self, name : str):
        return self.params[name]

    def progress(self) -> list[stage]:
        pass
