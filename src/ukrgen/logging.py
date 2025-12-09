# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from copy import deepcopy
import logging

def setup_loggers(debugall : bool =False, debug_systems : set[str] = list()):

    baselevel = logging.WARNING
    if debugall:
        baselevel = logging.DEBUG

    logging.basicConfig(level=logging.DEBUG)

    for name in ["TIF", "LSC", "LSCIRMOD",
                 "ADDR", "CODEGEN", "FNGEN",
                 "SCHED", "MRU", "BLISPATCH",
                 "patch_ng"]:
        logger = logging.getLogger(name)

        if name in debug_systems:
            print(f"{name} debug output enabled")
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(baselevel)

