# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import logging

from .composition import composition_stage
from ..gemm import gemm_context
from ..stage_param import stage_param

from ...schedulers.simple_dependency_scheduler import simple_dependency_scheduler

class lsc_schedule_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)


        self.debug = logging.getLogger("SCHED").debug

        for dep in ["rar","raw","war","waw"]:
            self.params[f"sched-{dep}-distance"] = stage_param(
                    value=0, 
                    description=f"Minimum distance for {dep} dependencies (in instructions)",
                    default=0,
                    required=False)

    def progress(self) -> list[composition_stage]:

        scheduler = simple_dependency_scheduler(
                rar=self.params["sched-rar-distance"].value,
                raw=self.params["sched-raw-distance"].value,
                war=self.params["sched-war-distance"].value,
                waw=self.params["sched-waw-distance"].value,
                debug_on=False)


        for dst,(targets,is_loop) in self.context.sched_map.items():
            src = []
            for bname in targets:
                src.extend(self.context.irs[bname])

            rescheduled = scheduler(src,loop=is_loop)

            self.context.irs[dst] = rescheduled

            self.debug(f"################### {dst.upper()} RE-SCHEDULED PSEUDO-ASM ###################")
            self.debug("\n".join(map(str,self.context.irs[dst])))


        self.context.params.update(self.params)

        return list()
