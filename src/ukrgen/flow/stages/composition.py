# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from __future__ import annotations

from ..ukr_context import ukr_context

from ..stage_param import stage_param

from .stage import stage

class composition_stage(stage):

    def __init__(self, context : ukr_context):
        pass

    def progress(self) -> list[composition_stage]:
        pass
