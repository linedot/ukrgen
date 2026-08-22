"""
Emulation for storing a single tile register in SME
"""


# enables signature:
#  - opdna1 1 tile with
#    - gp reg col offset, continuous rows
#    - gp reg row offset, continuous cols
#    - gp reg cols and rows
#    - vlen col offset, continuous rows
#    - vlen row offset, continuous cols
# emit (Option 1):
#    - rcreg <- 0
#    - loop:
#      - unroll i=1..n:
#        - store row/col rcreg+i of tile
#        - (advance address by col/row)
#      - rcreg += n
#      - cb (rcreg < svl, loop)
# emit (Option 2):
#    - rcreg <- 0
#    - loop:
#      - unroll i=1..n:
#        - extract row/col rcreg+i of tile into vreg_i
#        - store vreg_i into address
#        - (advance address by col/row)
#      - rcreg += n
#      - cb (rcreg < svl, loop)
# params: unroll = n
# allocations:
#   - Option 2: vreg * n
# irmods:
#  - fuse_tile_rc_loops
