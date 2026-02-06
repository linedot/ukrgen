# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from .composition import composition_stage
from ..gemm import gemm_context
from ..stage_param import stage_param

from ...schedulers.distance import distance_specification as dspec
from ...schedulers.simple_dependency_scheduler import simple_dependency_scheduler

import logging
from copy import deepcopy

class lsc_schedule_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)


        self.debug = logging.getLogger("SCHED").debug

        self.params["sched-distance-specs"] = stage_param(
                value=[], 
                description=("distance specifications for the scheduler"
                             " in \"dep:regt:rtag:op1:op2:dist\" format"),
                default=[],
                required=False,
                multi=True)

    def progress(self) -> list[composition_stage]:

        spec_strings = self.params["sched-distance-specs"].value

        if not spec_strings:
            for dst,(targets,is_loop) in self.context.sched_map.items():
                src = []
                for bname in targets:
                    src.extend(self.context.irs[bname])
                self.context.irs[dst] = deepcopy(src)
            self.context.params.update(self.params)
            return list()

        dsspecs = [dspec.from_string(s) for s in spec_strings]
        

        scheduler = simple_dependency_scheduler(dspecs=dsspecs)


        for dst,(targets,is_loop) in self.context.sched_map.items():
            src = []
            self.debug(f"Dependency re-scheduling {dst} from {','.join(targets)}")
            for bname in targets:
                src.extend(self.context.irs[bname])

            self.debug(f"################### {dst.upper()} UN-SCHEDULED PSEUDO-ASM ###################")
            for bname in targets:
                self.debug(f"======= {bname} =======")
                self.debug("\n".join(map(str,self.context.irs[bname])))

            rescheduled = scheduler(src,loop=is_loop)

            self.context.irs[dst] = rescheduled

            self.debug(f"################### {dst.upper()} RE-SCHEDULED PSEUDO-ASM ###################")
            self.debug("\n".join(map(str,self.context.irs[dst])))


        self.context.params.update(self.params)

        return list()
