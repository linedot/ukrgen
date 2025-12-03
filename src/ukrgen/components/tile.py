# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

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

def determine_dreg_tag(dima : dimension_properties, dimb : dimension_properties) -> str:

    dreg_tag = 'vreg'

    if dima.dt == dimension_type.vla and \
       dimb.dt == dimension_type.vla:
        dreg_tag = 'treg'
    elif dima.dt == dimension_type.fixed and dima.size > 1 and \
         dimb.dt == dimension_type.fixed and dimb.size > 1:
        dreg_tag = 'treg'
    # TODO: This would for example be SME vector register when
    #       using widening instructions (dot neighbours + outer product)
    #       i.e. the vector register stores a matrix, but is not
    #       a tile register. Make decision and document cleanly
    #       how we want to handle this
    #elif dima.dt == dimension_type.vla and \
    #     dimb.dt == dimension_type.fixed and dimb.size > 1:
    #    dreg_tag = 'treg'
    #elif dima.dt == dimension_type.fixed and dima.size > 1 \
    #     dimb.dt == dimension_type.vla:
    #    dreg_tag = 'treg'
    elif dima.dt == dimension_type.fixed and dima.size == 1 and \
         dimb.dt == dimension_type.fixed and dimb.size == 1:
        dreg_tag = 'freg'
    else:
        dreg_tag = 'vreg'

    return dreg_tag

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
                 stype : storage_type = storage_type.register,
                 bands : tuple[int,int] = (-1,-1)):
        self.storage_type = storage_type
        self.dima = dima
        self.dimb = dimb
        self.subtiles = subtiles
        self.subtile_count_a = subtile_count_a
        self.subtile_count_b = subtile_count_b
        self.bands = bands

    def check_zero(self, idx : tuple[int,int]) -> bool:

        if self.bands[0] != -1 and idx[0] > (idx[1] + self.bands[0]):
            return True
        if self.bands[1] != -1 and idx[1] > (idx[0] + self.bands[1]):
            return True

        return False


    @property
    def is_scalar(self) -> bool:
        return self.dima.dt == dimension_type.fixed and \
               self.dima.size == 1 and \
               self.dimb.dt == dimension_type.fixed and \
               self.dimb.size == 1

    @property
    def is_vla_vector(self) -> bool:

        return (self.dima.dt == dimension_type.vla and \
                self.dima.size == 1 and \
                self.dimb.dt == dimension_type.fixed and \
                self.dimb.size == 1) or \
               (self.dima.dt == dimension_type.fixed and \
                self.dima.size == 1 and \
                self.dimb.dt == dimension_type.vla and \
                self.dimb.size == 1)

    @property
    def is_fixed_vector(self) -> bool:

        return (self.dima.dt == dimension_type.fixed and \
                self.dimb.dt == dimension_type.fixed) and \
               ((self.dima.size == 1) != \
                (self.dimb.size == 1))

    @property
    def is_vector(self) -> bool:
        return self.is_vla_vector or self.is_fixed_vector

    @property
    def is_vla_tile(self) -> bool:
        return (self.dima.dt == dimension_type.vla and \
                self.dima.size == 1 and \
                self.dimb.dt == dimension_type.vla and \
                self.dimb.size == 1)

    @property
    def is_fixed_tile(self) -> bool:

        return (self.dima.dt == dimension_type.fixed and \
                self.dimb.dt == dimension_type.fixed) and \
               ((self.dima.size > 1) and \
                (self.dimb.size > 1))

    @property
    def is_tile(self) -> bool:
        return self.is_vla_tile or self.is_fixed_tile

    def __str__(self):
        sdt = lambda dt : "V" if dt==dimension_type.vla else ""

        return f"{self.dima.size}{sdt(self.dima.dt)}X{self.dimb.size}{sdt(self.dimb.dt)}"

    def __repr__(self):
        return self.__str__()



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
                 subdims : tuple[dimension_properties,dimension_properties],
                 bands : tuple[int,int] = (-1,-1)):

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
                stype = storage_type.register,
                bands = bands)

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
