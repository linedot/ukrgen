"""
Resolutions for SCALAR_REDUCE transformation
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


class scalar_reduce_provider(resolution_provider):
    """
    Provides resolutions for a SCALAR_REDUCE vector operand
    """
    def register_resolutions(self, registry : resolution_registry):

        # example RVV:
        #  fld f0, (t0)
        #  vfmacc.vf v0, f0, v1
        registry.add_resolution(
            key=tfr_key(tfs=frozenset({tf.SCALAR_REDUCE}),
                        rtype=rgt.VEC, ddir=dmd.IN),
            rsln=tr(
                unique_tag='use_scalar',
                opd_mod_reqs={omod.VF},
                steps=[
                    dm_step(op="load", dest=orig_ref(), dest_rtype=rgt.FP)
                    ]))

        # example SVE:
        #  fld d0, [x5]
        #  dup z1.d, z0.d[0]
        #  fmla z2.d, p0/m, z3.d, z1.d
        registry.add_resolution(
            key=tfr_key(tfs=frozenset({tf.SCALAR_REDUCE}),
                        rtype=rgt.VEC, ddir=dmd.IN),
            rsln=tr(
                unique_tag='bc_scalar',
                steps=[
                   dm_step(op="load", dest=temp_ref(tag="0"), dest_rtype=rgt.FP),
                   dm_step(op="move", dest=orig_ref(), dest_rtype=rgt.VEC,
                           src=[temp_ref(tag="0")],src_rtypes=[rgt.FP],
                           opd_mod_reqs={orig_ref(): {omod.BCAST}})
                   ]))

        # example SVE:
        #  ld1rd z0.d, p0/m, [x5]
        #  fmla z2.d, p0/m, z3.d, z1.d
        registry.add_resolution(
            key=tfr_key(tfs=frozenset({tf.SCALAR_REDUCE}),
                        rtype=rgt.VEC, ddir=dmd.IN),
            rsln=tr(
                unique_tag='bc_load',
                steps=[
                    dm_step(op="load", dest=orig_ref(), dest_rtype=rgt.VEC,
                            opd_mod_reqs={orig_ref(): {omod.BCAST}})
                    ]))

        # example NEON:
        #  ld1 v0.d[1], [x5]
        #  dup v1.2d, v0.d[1]
        #  fmla v2.2d, v3.2d, v1.2d
        registry.add_resolution(
            key=tfr_key(tfs=frozenset({tf.SCALAR_REDUCE}),
                        rtype=rgt.VEC, ddir=dmd.IN),
            rsln=tr(
                unique_tag='bc_lane',
                steps=[
                    dm_step(op="load", dest=temp_ref(tag="0"), dest_rtype=rgt.VEC,
                            opd_mod_reqs={temp_ref(tag="0") : {omod.ILANE}}),
                    dm_step(op="move", dest=orig_ref(), dest_rtype=rgt.VEC,
                            src=[temp_ref(tag="0")],src_rtypes=[rgt.VEC],
                            opd_mod_reqs={
                                temp_ref(tag="0") : {omod.ILANE},
                                orig_ref(): {omod.BCAST}
                                })
                    ]))

        # example NEON:
        # ld1 v0.d[1], [x5]
        # fmla v2.2d, v3.2d, v1.d[1]
        registry.add_resolution(
            key=tfr_key(tfs=frozenset({tf.SCALAR_REDUCE}),
                        rtype=rgt.VEC, ddir=dmd.IN),
            rsln=tr(
                unique_tag='use_lane',
                opd_mod_reqs={omod.ILANE},
                steps=[
                    dm_step(op="load", dest=orig_ref(), dest_rtype=rgt.VEC,
                            opd_mod_reqs={orig_ref() : {omod.ILANE}}),
                    ]))

        # example neon:
        # ldr d0, [x5]
        # fmov v1.d[1], d0
        # fmla v2.2d, v3.2d, v1.d[1]
        registry.add_resolution(
            key=tfr_key(tfs=frozenset({tf.SCALAR_REDUCE}),
                        rtype=rgt.VEC, ddir=dmd.IN),
            rsln=tr(
                unique_tag='scalar_to_lane',
                opd_mod_reqs={omod.ILANE},
                steps=[
                   dm_step(op="load", dest=temp_ref(tag="0"), dest_rtype=rgt.FP),
                   dm_step(op="move", dest=orig_ref(), dest_rtype=rgt.VEC,
                           src=[temp_ref(tag="0")],src_rtypes=[rgt.FP],
                           opd_mod_reqs={orig_ref(): {omod.ILANE}})
                   ]))

        # STORE:


        # if the output is a scalar we can just store a scalar?
        registry.add_resolution(
            key=tfr_key(tfs=frozenset({tf.SCALAR_REDUCE}),
                        rtype=rgt.VEC, ddir=dmd.OUT),
            rsln = tr(
                unique_tag='use_scalar',
                opd_mod_reqs={omod.VF},
                steps=[
                    dm_step(op="store", src=[orig_ref()], src_rtypes=[rgt.FP]),
                    ]))


        # example NEON:
        # st1 v0.d[1], [x5]
        registry.add_resolution(
            key=tfr_key(tfs=frozenset({tf.SCALAR_REDUCE}),
                        rtype=rgt.VEC, ddir=dmd.OUT),
            rsln = tr(
                unique_tag='use_lane',
                # We can probably weaken the requirement without breaking correctness,
                # Allowing for weird stuff like fma(bcast:a, bcast:b, bcast:c) -> store_lane:c
                #opd_mod_reqs={omod.ILANE},
                steps=[
                    dm_step(op="store", src=[orig_ref()], src_rtypes=[rgt.VEC],
                            opd_mod_reqs={orig_ref() : {omod.ILANE}}),
                    ]))


        # example NEON:
        # fmov d1, v0.d[1]
        # str d1, [x5]
        registry.add_resolution(
            key=tfr_key(tfs=frozenset({tf.SCALAR_REDUCE}),
                        rtype=rgt.VEC, ddir=dmd.OUT),
            rsln = tr(
                unique_tag='lane_to_scalar',
                #opd_mod_reqs={omod.ILANE},
                steps=[dm_step(op="move", dest=temp_ref(tag='0'), dest_rtype=rgt.FP,
                               src=[orig_ref()], src_rtypes=[rgt.VEC],
                               opd_mod_reqs={orig_ref(): {omod.ILANE}}),
                       dm_step(op="store", src=[temp_ref(tag='0')], src_rtypes=[rgt.FP])]))
