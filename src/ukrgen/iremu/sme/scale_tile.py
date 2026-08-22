"""
Emulation for scaling a single tile register in SME
"""


# enables signature:
#  - fmul/opd3 (tile1,scalar,tile2)
# emit:
#    - rcreg <- 0
#    - loop:
#      - unroll i=1..n:
#        - extract row/col rcreg+i of tile1 into vreg_i
#        - fmul vreg_i, factor, vreg_i
#        - insert vreg_i into row/col rcreg+i of tile2
#      - rcreg += n
#      - cb (rcreg )
# params: unrolls = n
# allocations: vreg * n
# irmods:
#  - fuse_tile_rc_loops
#  - fuse_tile_rc_scale_tma
#  - fuse_tile_rc_scale_ldst
