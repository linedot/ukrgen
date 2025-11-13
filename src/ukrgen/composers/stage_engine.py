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
                 get_param_callback : Callable[[composition_stage,str],str]):

        self.stages = stages
        self.ctx = ctx
        self.get_param_callback = get_param_callback


    def run(self):
        for stage_ctr in self.stages:
            stage_deque = deque([stage_ctr])
            while stage_deque:
                ctr = stage_deque.popleft()
                stage  = ctr(context=self.ctx)
                for param in stage.get_parameter_names():
                    val = None
                    if param not in self.ctx.params:
                        val = self.get_param_callback(stage,param)
                    else:
                        val = self.ctx.params[param]
                    stage.set_param(param, val)

                stage_deque.extend(stage.progress())
            

