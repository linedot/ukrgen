"""
Classes and datatypes for querying if a capability is
supported in an asmgen generator
"""

from itertools import combinations
from dataclasses import dataclass

from asmgen.asmblocks.noarch import asmgen
from asmgen.registers import adt_triple, asm_data_type as adt
from asmgen.asmblocks.op import (
    operation,
    opd3,
    opd3_modifier,
    opdna1,
    opdna1_modifier,
    operand_shape,
    operand_type as otype,
    register_type as rtype,
    operand_constraint as opcst,
    operation_signature as opsig
)

from ..matching.math import (
    ast_node,
    solve_requirement,
    for_each_expression,
    HW_FADD_AST,
    HW_FMUL_AST,
    HW_FMA_AST,
    HW_FDOTA_AST,
    HW_FOPA_AST,
    HW_MMA_AST
)

from ..components.tile import (
    dimension_type as dimt,
    dimension_properties as dimp,
    tile
)
from ..models.lsc.reg import lsc_reg_type as lrt

def infer_hw_tile(gen : asmgen,
                  shape : operand_shape,
                  dt : adt) -> tile:
    """
    Determine operand hw tile from operand shape and 

    :param gen: generator to query for additional information
    :param shape: shape of the operand
    :param dt: element data type

    :return: 2D tile representation of the operand
    """
        

    vla_vec_dp = dimp(dt=dimt.vla, size=1,
                      sdt=dimt.vla, sd_size=1)
    fixed_scalar_dp = dimp(dt=dimt.fixed, size=1,
                           sdt=dimt.fixed, sd_size=1)

    # an immediate is a scalar tile
    if otype.IMMEDIATE == shape.otype:
        return tile(dima=fixed_scalar_dp,
                    dimb=fixed_scalar_dp)

    if otype.REGISTER == shape.otype and \
            shape.rtype in {rtype.GP,rtype.FP,rtype.MASK}:
                return tile(dima=fixed_scalar_dp,
                            dimb=fixed_scalar_dp)

    is_vla = gen.is_vla

    vec_dp = vla_vec_dp

    if not is_vla:
        element_count = gen.simd_size//adt_size(dt)
        #TODO: figure out from signatures how many elements are
        #      subindexable (AVX can only do the first 128bits)
        vec_dp = dimp(dt=dimt.fixed, size=element_count,
                      sdt=dimt.fixed, sd_size=element_count)

    #TODO: figure out how to support BLOCKIDX (SVE), is there a need to
    #      encode more info in 'tile'? Or should that be handled some
    #      other way?

    if otype.REGISTER == shape.otype and \
            shape.rtype in {rtype.VEC}:
        return tile(vec_dp,fixed_scalar_dp)


    # Just VLA in both dimensions. Potentially different actual sizes
    tile_tile = tile(vla_vec_dp,vla_vec_dp)
    
    # If dims are VLA but a ratio is known, could be something like this?
    # tile_1r2c = tile(dimp(dimt.vla, 1, dimt.vla, 1),
    #                  dimp(dimt.vla, 2, dimt.vla, 2)
    
    # it might make sense to encode whether or not the 2 dimensions are
    # equal or not in the 'tile' structures when both are VLA

    if not is_vla:
        # TODO: figure out how and where to incode information about rows
        #       and columns
        nrows = gen.tile_rows
        ncols = gen.tile_cols
        tile_tile = tile(dima=dimp(dt=dimt.fixed, size=nrows,
                                   sdt=dimt.fixed, sd_size=nrows),
                         dimb=dimp(dt=dimt.fixed, size=ncols,
                                   sdt=dimt.fixed, sd_size=ncols),
                         )

    # TODO: One dim being VLA and the other being fixed is also possible,
    #       figure out how to handle it

    if otype.REGISTER == shape.otype and \
            shape.rtype in {rtype.TILE}:
        return tile_tile

    raise NotImplementedError("Operand shape not supported")

class op_support:
    """
    Support for an operation with specific operands
    """
    def __init__(self, signature : opsig):

        self.signature = signature

        # data_tile
        self.data_tiles : dict[str, tile] = {}
        self.hw_tiles : dict[str,tile] = {}

        for name,shape in self.signature.operands.items():
            self.hw_tiles[name] = infer_hw_tile(shape)


def generate_op_supports(gen : asmgen,
                         signature : opsig) -> list[op_support]:
    hw_tiles = {
        name : infer_hw_tile(gen,shape,shape.dt) \
            for name,shape in signature.operands.items()
    }
    supports = []

    

class op_support_builder:

    op_base_asts = {
        'fadd': HW_FADD_AST,
        'fmul': HW_FMUL_AST,
        'fma': HW_FMA_AST,
        'fdota': HW_FDOTA_AST,
        'fopa': HW_FOPA_AST,
        'mma': HW_MMA_AST
            }

    def __init__(self, gen : asmgen):
        self.gen = gen

    
    @classmethod
    def get_base_op(cls, ast : ast_node):
        for name,opast in cls.op_base_asts:
            if opast == ast:
                return name

        raise ValueError(f"Op not found for AST: {ast}")


    def determine_base_support(self):

        # Is this VLA
        self.is_vla = self.gen.is_vla
        self.fregs_in_vregs = self.gen.are_fregs_in_vregs
        
        # TODO: figure out through signatures
        self.mregs_are_vregs = False

        arith_ops = ['fadd','fmul',
                     'fma',
                     'fdota',
                     'fopa',
                     'mmul',
                     'madd',
                     'mma','msa']

        self.arith_ops = [op for op in arith_ops if hasattr(self.gen, op)]

        # The generator is guaranteed to have those, but for consistencies sake...
        ldst_ops = ['load','store']
        self.ldst_ops = [op for op in ldst_ops if hasattr(self.gen,op)]
    

    def create_base_solutions(self, req : ast_node) -> list[dict]:
        available_asts = [op_base_asts[op] for op in self.arith_ops]

        return solve_requirement(req, available_asts)

    def determine_scalarization_options(self,
                                        ast : ast_node,
                                        opsigs : list[operation_signature],
                                        osh: operand_shape) -> list[dict]:
        
        # if input: BCAST, VF, LANE
        # if output: LANE

        for_each_expression(ast, func)

        pass


    def adapt_solutions_to_isa(self, solutions : [list[dict]]) -> list[dict]:
        isa_solutions = []

        for solution in solutions:
            # A solution is a list of ASTs
            for smap in solution:
                #TODO:
                ast = smap['hw_ast']
                opname = self.get_base_op(ast)
                op : operation = getattr(self.gen,opname)

                sigs = op.get_signatures()
        pass
