# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from typing import Type,Callable
from collections import deque

from .stages.composition import composition_stage

#TODO: generalized ukr context
from .gemm import gemm_context


class stage_engine:
    def __init__(self, stages : list[Type[composition_stage]],
                 ctx : gemm_context,
                 prolog : Callable[[composition_stage],None] = lambda s : None,
                 epilog : Callable[[composition_stage],None] = lambda s : None):

        self.stages = stages
        self.ctx = ctx
        self.prolog = prolog
        self.epilog = epilog


    def run(self):
        for stage_ctr in self.stages:
            stage_deque = deque([stage_ctr])
            while stage_deque:
                ctr = stage_deque.popleft()
                stage  = ctr(context=self.ctx)

                self.prolog(stage)

                # check parameters
                for pname in stage.get_parameter_names():
                    if stage.params[pname].value is None and \
                            stage.params[pname].required:
                        raise ValueError(f"Parameter {pname} is required!")

                stage_deque.extend(stage.progress())
                self.epilog(stage)
