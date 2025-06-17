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


scalar = dimension_properties(dt=dimension_type.fixed, size=1,
                              sdt=dimension_type.fixed, sd_size=1)

vla_vector = dimension_properties(dt=dimension_type.vla, size=1,
                                  sdt=dimension_type.fixed, sd_size=4)

x4_vector = dimension_properties(dt=dimension_type.fixed, size=4,
                                 sdt=dimension_type.fixed, sd_size=4)

class simple_ukr_tile(tile):
    def __init__(self, 
                 a_size : int,
                 b_size : int,
                 subdims : tuple[dimension_properties,dimension_properties]):

        dima = dimension_properties(dt=dimension_type.fixed,
                                    size=a_size,
                                    sdt=dimension_type.fixed,
                                    sd_size=a_size)
        dimb = dimension_properties(dt=dimension_type.fixed,
                                    size=b_size,
                                    sdt=dimension_type.fixed,
                                    sd_size=b_size)

        super().__init__(
                dima=dima,dimb=dimb,
                subtiles = [tile(dima=subdims[0],dimb=subdims[1])],
                subtile_count_a = 1,
                subtile_count_b = 1,
                stype = storage_type.register)

class composed_ukr_tile(tile):
    def __init__(self, 
                 a_sizes : list[int],
                 b_sizes : list[int],
                 subdims : tuple[dimension_properties,dimension_properties]):

        tiles = [simple_ukr_tile(a_size=a, b_size=b, subdims=subdims) for a in a_sizes for b in b_sizes]

        dima = dimension_properties(dt=dimension_type.fixed,
                                    size=len(a_sizes),
                                    sdt=dimension_type.fixed,
                                    sd_size=len(a_sizes))
        dimb = dimension_properties(dt=dimension_type.fixed,
                                    size=len(b_sizes),
                                    sdt=dimension_type.fixed,
                                    sd_size=len(b_sizes))
        super().__init__(
                dima=dima,dimb=dimb,
                subtiles =tiles,
                subtile_count_a = len(a_sizes),
                subtile_count_b = len(b_sizes),
                stype = storage_type.memory)
