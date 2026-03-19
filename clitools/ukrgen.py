# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import logging
import sys

from argparse import ArgumentParser

from ukrgen.flow.stages.stage import stage
from ukrgen.flow.stages.support import support_stage
from ukrgen.flow.stages.datatype import datatype_stage
from ukrgen.flow.stages.dimension import dimension_stage
from ukrgen.flow.stages.tif import mm_tif_stage
from ukrgen.flow.stages.model import lsc_model_stage
from ukrgen.flow.stages.mru import lsc_mru_stage
from ukrgen.flow.stages.schedule import lsc_schedule_stage
from ukrgen.flow.stages.specialize import specialize_lsc_stage
from ukrgen.flow.stages.irmod import irmod_inserter_stage
from ukrgen.flow.stages.codegen import blis_ukr_codegen_stage

from ukrgen.flow.ukr_context import ukr_context
from ukrgen.flow.stage_engine import stage_engine

from ukrgen.logging import setup_loggers

helpargs=['-h','--help']

def helpexit_if_last_parser(rest : list[str], parser : ArgumentParser):
    if any(a in rest for a in helpargs):
        if not [a for a in rest if a not in helpargs]:
            parser.print_help()
            sys.exit(0)

class argparse_prolog:
    def __init__(self, parser : ArgumentParser):
        self.parser=parser
        self.rest=list()

    def __call__(self, s: stage):
        params = s.get_parameter_names()

        if not params:
            return

        for pname in params:
            self.parser.add_argument(
                f"--{pname}",
                nargs="+" if s.get_param(pname).multi else None,
                help=s.get_param(pname).description,
                default=s.get_param(pname).default,
                choices=s.get_param(pname).choices,
                required=s.get_param(pname).required)


        args, self.rest = self.parser.parse_known_args()
        helpexit_if_last_parser(rest=self.rest, parser=self.parser)

        for pname in params:

            apname = pname.replace("-", "_")
            val = args.__dict__[apname]

            s.set_param(pname, val)


def ukrgen():

    ukr_ctx = ukr_context()

    parser = ArgumentParser(add_help=False)

    parser.add_argument("--debug",
                        help="Enables debug output for a specific subsystem; without any arguments, all debug output is enabled",
                        nargs="*",
                        type=str)

    args,rest = parser.parse_known_args()

    debugall = False
    debug_systems = list()

    if args.debug is not None:
        if len(args.debug) > 0:
            debug_systems = set(args.debug)
        else:
            debugall = True

    setup_loggers(debugall=debugall, debug_systems=debug_systems)

    stages = [
        support_stage,
        datatype_stage,
        dimension_stage,
        mm_tif_stage,
        lsc_model_stage,
        specialize_lsc_stage,
        irmod_inserter_stage,
        lsc_mru_stage,
        lsc_schedule_stage,
        blis_ukr_codegen_stage]

    prolog = argparse_prolog(parser=parser)

    se = stage_engine(stages=stages,
                      ctx=ukr_ctx,
                      prolog=prolog)

    se.run()

    # prolog won't print help if there are any parameters not consumed
    # by stages
    if any(a in prolog.rest for a in helpargs):
        parser.print_help()
        sys.exit(0)

    print(ukr_ctx.asmblocks["full_function"])

if __name__ == "__main__":
    ukrgen()
