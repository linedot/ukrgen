"""
Resolutions for direct loads/stores
"""

from asmgen.asmblocks.op import (
    register_type as rgt,
    operand_modifier as omod,
)

from ...matching.math import transformation as tf

from ..data_move import (
    resolution_registry,
    tfr_key,
    transformation_resolution as tr,
    dm_direction as dmd,
    dm_step,
    orig_ref,
    temp_ref,
    resolution_provider
)


class direct_provider(resolution_provider):
    """
    Provides resolutions for direct loads/stores
    """
    def register_resolutions(self, registry : resolution_registry):


        for rtype in rgt:

            # Direct load
            registry.add_resolution(
                key=tfr_key(tfs=frozenset(),
                            rtype=rtype, ddir=dmd.IN),
                rsln=tr(
                    unique_tag='direct',
                    steps=[
                        dm_step(op="load", dest=orig_ref(), dest_rtype=rtype)
                        ]))

            # Direct store
            registry.add_resolution(
                key=tfr_key(tfs=frozenset(),
                            rtype=rtype, ddir=dmd.OUT),
                rsln=tr(
                    unique_tag='direct',
                    steps=[
                        dm_step(op="store", dest=orig_ref(), dest_rtype=rtype)
                        ]))
            
            # Exactly the same, but with tf.NONE instead of an empty set

            # Direct load
            registry.add_resolution(
                key=tfr_key(tfs=frozenset({tf.NONE}),
                            rtype=rtype, ddir=dmd.IN),
                rsln=tr(
                    unique_tag='direct',
                    steps=[
                        dm_step(op="load", dest=orig_ref(), dest_rtype=rtype)
                        ]))

            # Direct store
            registry.add_resolution(
                key=tfr_key(tfs=frozenset({tf.NONE}),
                            rtype=rtype, ddir=dmd.OUT),
                rsln=tr(
                    unique_tag='direct',
                    steps=[
                        dm_step(op="store", dest=orig_ref(), dest_rtype=rtype)
                        ]))
