# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest

from ukrgen.components import (
        dimension_properties,
        dimension_type,
        simple_ukr_tile,
        storage_type,
        tile,
        scalar_dp,
        scalar_tile,
        vla_vector,
        x4_vector
        )
from ukrgen.generators import mm,order2D
from ukrgen.models import load_store_cpu,addr_resolver
from ukrgen.models.load_store_operations import lsc_offset
from ukrgen.models.tile_offset_mapper import flat_mapper

from ukrgen.specializers.asm import lsc_specializer


from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.sve import sve

from asmgen.registers import reg_tracker


class test_lsc_specializer(unittest.TestCase):
    #TODO: specializer tests
    pass
