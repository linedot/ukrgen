"""
Tests for data movement resolution
"""
import unittest

from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.op import register_type as rgt
from asmgen.registers import asm_data_type as adt

from ukrgen.matching.math import transformation as tf
from ukrgen.support.dm_resolutions.scalar_reduce import scalar_reduce_provider
from ukrgen.support.data_move import (
    resolution_registry,
    resolution_provider,
    tfr_key,
    dm_direction as dmd,
    dm_step,
    transformation_resolution as tr,
    orig_ref,
    check_resolution
)


class none_provider(resolution_provider):
    """
    Provides base data movement for operands that require NO transformation.
    """
    def register_resolutions(self, registry: resolution_registry):
        # Base load (IN)
        registry.add_resolution(
            key=tfr_key(tfs=frozenset({tf.NONE}),
                        rtype=rgt.VEC, ddir=dmd.IN),
            rsln=tr(
                unique_tag="direct",
                steps=[
                    dm_step(op="load", dest=orig_ref(), dest_rtype=rgt.VEC)
                ]
            )
        )
        # Base store (OUT)
        registry.add_resolution(
            key=tfr_key(tfs=frozenset({tf.NONE}),
                        rtype=rgt.VEC, ddir=dmd.OUT),
            rsln=tr(
                unique_tag="direct",
                steps=[
                    dm_step(op="store", src=[orig_ref()], src_rtypes=[rgt.VEC])
                ]
            )
        )


class test_resolution_checker(unittest.TestCase):
    """
    Testsuite for the transformation resolution checker
    """
    def setUp(self):
        self.gen = sve()
        self.registry = resolution_registry()

        # Register providers
        scalar_reduce_provider().register_resolutions(self.registry)
        none_provider().register_resolutions(self.registry)

    def test_solution_0_plausibility(self):
        """
        Tests if Solution 0 is plausible on SVE hardware.
        Solution 0: bdreg -> SCALAR_REDUCE, cdreg -> NONE, adreg -> NONE
        """

        # Extracted directly from your print_solution output
        transformations = {
            'cdreg': [tf.NONE],
            'bdreg': [tf.SCALAR_REDUCE],
            'adreg': [tf.NONE]
        }

        target_op = "fma"
        target_dt = adt.FP32  # Test for standard 32-bit floats

        # In FMA, cdreg acts as both IN and OUT. We'll check the IN path here.
        directions = {
            'cdreg': dmd.IN,
            'bdreg': dmd.IN,
            'adreg': dmd.IN
        }

        # Verify every operand in the solution
        for opd_name, tfs in transformations.items():
            key = tfr_key(tfs=frozenset(tfs), rtype=rgt.VEC, ddir=directions[opd_name])

            candidates = self.registry.tfr_map.get(key, [])
            self.assertTrue(len(candidates) > 0, f"No resolutions registered for {key}")

            valid_resolution_found = False
            for rsln in candidates:
                # Check if this specific hardware supports this resolution path
                if check_resolution(self.gen, rsln, opd_name, target_dt, target_op):
                    valid_resolution_found = True
                    break

            self.assertTrue(
                valid_resolution_found,
                f"Hardware does not support operand '{opd_name}' with transforms {tfs}"
            )
