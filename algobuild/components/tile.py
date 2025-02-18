# annotation for self type
from __future__ import annotations

from enum import Enum,auto

class storage_type(Enum):
    register = auto()
    memory = auto()

class dimension_type(Enum):
    fixed = auto()
    vla = auto()
    # I think we don't need this, and fixed fits the functionality
    # subtile = auto()

class dimension_properties:
    def __init__(self, dt : dimension_type, size : int, sdt : dimension_type, sd_size : int):
        self.dt = dt
        self.size = size
        self.sdt = sdt
        self.sd_size = sd_size


class tile:
    def __init__(self, 
                 dima : dimension_properties, # m for c, m for a, k for b
                 dimb : dimension_properties, # n for c, k for a, n for b
                 # 3D tiles? 3D registers? Tensors? :D maybe in the future
                 # dims : list[dimension_properties]
                 # Thinking of tiling with differently-sized kernels in 2D...
                 subtiles : list[tile] = None,
                 subtile_count_a : int = None,
                 subtile_count_b : int = None,
                 # ND?
                 # subtile_counts : list[int] = None,
                 stype : storage_type = storage_type.register):
        self.storage_type = storage_type
        self.dima = dima
        self.dimb = dimb
        self.subtiles = subtiles
        self.subtile_count_a = subtile_count_a
        self.subtile_count_b = subtile_count_b



scalar_dp = dimension_properties(dt=dimension_type.fixed, size=1,
                                 sdt=dimension_type.fixed, sd_size=1)

scalar_tile = tile(dima=scalar_dp, dimb=scalar_dp)
