"""
Emulation for scaling a single tile register and adding it to another one in SME
"""


# enables signature:
#  - tma/opd3 (tile1,scalar,tile2)
# emit:
#    - rcreg <- 0
#    - loop:
#      - unroll i=1..n:
#        - extract row/col rcreg+i of tile1 into vreg_2i
#        - extract row/col rcreg+i of tile2 into vreg_2i+1
#        - fma vreg_2i, factor, vreg_2i+1
#        - insert vreg_2i+1 into row/col rcreg+i of tile2
#      - rcreg += n
#      - cb (rcreg )
# params: unrolls = n
# allocations: vreg * n * 2
# irmods:
#  - fuse_tile_rc_loops
#  - fuse_tile_rc_tma_scale
#  - fuse_tile_rc_tma_ldst
