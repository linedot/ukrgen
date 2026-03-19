# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from typing import Type,Callable
from collections import deque

from .stages.stage import stage

from .ukr_context import ukr_context


class stage_engine:
    def __init__(self, stages : list[Type[stage]],
                 ctx : ukr_context,
                 prolog : Callable[[stage],None] = lambda s : None,
                 epilog : Callable[[stage],None] = lambda s : None):

        self.stages = stages
        self.ctx = ctx
        self.prolog = prolog
        self.epilog = epilog


    def run(self):
        for stage_ctr in self.stages:
            stage_deque = deque([stage_ctr])
            while stage_deque:
                ctr = stage_deque.popleft()
                s  = ctr(context=self.ctx)

                self.prolog(s)

                # check parameters
                for pname in s.get_parameter_names():
                    if s.params[pname].value is None and \
                            s.params[pname].required:
                        raise ValueError(f"Parameter {pname} is required!")

                stage_deque.extend(s.progress())
                self.epilog(s)
