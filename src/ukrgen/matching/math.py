"""
Matching of mathematical operations
"""
from __future__ import annotations

import logging
import itertools
from copy import deepcopy,copy
from typing import Optional,Iterator
from dataclasses import dataclass
from enum import Enum,auto

class operation(Enum):
    """
    Mathematical operations
    """
    ADD = auto()
    MUL = auto()
    REDUCE_SUM = auto()
    MOVE = auto()


class transformation(Enum):
    """
    Tile transformations
    """
    NONE          = auto()
    TRANSPOSE     = auto() # (i,j)    -> (j,i)
    COL_REDUCE    = auto() # (i,j)    -> (None,j)
    ROW_REDUCE    = auto() # (i,j)    -> (i,None)
    SCALAR_REDUCE = auto() # (i,None)/(None,j) -> (None,None) [BCAST/VF/IDX/LANE]

debug = logging.getLogger("MATCHING").debug

@dataclass(frozen=True)
class operand_ref:
    """
    Reference to an operand
    """
    name: str
    indices: tuple[str,...]

    def get_shape(self) -> list[str]:
        """
        Return the set of active dimensions
        """
        return list(self.indices)

        #return set(idx for idx in self.indices if idx is not None)

    def is_dimensionally_valid(self) -> bool:
        """
        Operands are always valid
        """
        return True

    def __str__(self):
        idx_str = ','.join(i if i is not None else '1'
                           for i in self.indices )
        return f"{self.name}[{idx_str}]"

    def __repr__(self):
        return str(self)

@dataclass(frozen=True)
class expression_node:
    """
    Node representing an expression on mathematical operands
    """
    op: operation
    left: expression_node|operand_ref
    right: Optional[expression_node|operand_ref] = None
    reduce_dim: Optional[str] = None

    def get_shape(self) -> list[str]:
        """
        Return the output shape of the expression
        """

        # Potential OOP rework required if more ops are added in the future
        if self.op in (operation.ADD, operation.MUL):
            left_indices = self.left.get_shape()
            if self.right is None:
                return left_indices

            right_indices = self.right.get_shape()

            s = []
            maxlen = max(len(left_indices), len(right_indices))

            l_pad = left_indices + [None] * (maxlen - len(left_indices))
            r_pad = right_indices + [None] * (maxlen - len(right_indices))

            for l,r in zip(l_pad,r_pad):
                s.append(l if l is not None else r)
            return s

        elif self.op == operation.REDUCE_SUM:

            return [i for i in self.left.get_shape() if i != self.reduce_dim]
        elif self.op == operation.MOVE:
            return self.left.get_shape()
        return []

    def is_dimensionally_valid(self) -> bool:
        """
        Checks if all assignments inside the AST are valid
        """

        if not self.left.is_dimensionally_valid():
            return False

        if self.right is not None:
            if not self.right.is_dimensionally_valid():
                return False


        # Forbid vector->scalar
        # Allow scalar->vector
        if self.op == operation.MOVE:
            left_shape = self.left.get_shape()
            # right is not None for a MOVE
            right_shape = self.right.get_shape()

            for i_in,i_out in zip(right_shape, left_shape):

                # Vector -> scalar
                if i_out is None and i_in is not None:
                    return False
                # index mismatch
                if i_out is not None and i_in is not None:
                    if i_out != i_in:
                        return False
        return True

    def __str__(self):
        maybe_right = ""
        if self.right is not None:
            maybe_right = f",{self.right}"

        if operation.REDUCE_SUM == self.op:
            maybe_right += f",{self.reduce_dim}"
        return f"{self.op.name}({self.left}{maybe_right})"

    def __repr__(self):
        return str(self)


type ast_node = expression_node | operand_ref

def recurse_indices(n : ast_node) -> list[str,...]:
    """
    Gathers all indices of an AST node and it's children, preserving order

    :param n: expression node or a leaf operand reference
    :return: set of all indices in the node and all it's children
    """
    if isinstance(n, operand_ref):
        return [idx for idx in n.indices if idx is not None]

    indices = recurse_indices(n.left)
    if n.right is not None:
        for idx in recurse_indices(n.right):
            if idx not in indices:
                indices.append(idx)

    # remove reduction index if this op is a reduction
    if operation.REDUCE_SUM == n.op:
        indices = [idx for idx in indices if idx != n.reduce_dim]

    return indices

def simplify_ast(n : ast_node) \
        -> ast_node:
    """
    Simplify AST

    :param n: starting node of the AST
    :return: simplified AST
    """

    if isinstance(n, operand_ref):
        return n

    new_left = simplify_ast(n.left)
    new_right = None
    if n.right is not None:
        new_right = simplify_ast(n.right)

    # For now only a reducing sum is relevant
    # FMA (VF transform) -> already scalar
    # FDOTA,FOPA,MMA -> k reduced
    if operation.REDUCE_SUM == n.op:
        child_indices = recurse_indices(new_left)
        if new_right is not None:
            child_indices |= recurse_indices(new_right)

        # If the index isn't present in the children, REDUCE_SUM
        # doesn't do anything
        if n.reduce_dim not in child_indices:
            return new_left

    return expression_node(
            op=n.op,
            left=new_left, right=new_right,
            reduce_dim=n.reduce_dim)

def map_and_match(requirement : ast_node,
                  hardware : ast_node,
                  name_mapping : dict[str,str],
                  index_mapping : dict[str,str]) -> bool:
    """
    Maps the indices between the requirement and hardware and checks whether
    the hw AST matches the required AST
    """

    debug(f"Trying to match {requirement} and {hardware}")

    if type(requirement) != type(hardware):
        debug(f"mismatch: different node types")
        return False

    if isinstance(requirement, operand_ref):
        if hardware.name not in name_mapping:
            name_mapping[hardware.name] = requirement.name
        elif requirement.name != name_mapping[hardware.name]:
            debug((f"mismatch: {hardware.name} already mapped "
                   f"to {name_mapping[hardware.name]}, not {requirement.name}"))
            return False

        if len(requirement.indices) != len(hardware.indices):
            debug(f"mismatch: index length between {requirement} and {hardware}")
            return False

        for r_idx, h_idx in zip(requirement.indices, hardware.indices):
            if h_idx is None:
                continue
            if h_idx in index_mapping:
                if index_mapping[h_idx] != r_idx:
                    debug((f"mismatch: index mapping in {requirement} and {hardware}:"
                           f" {h_idx} is mapped to {index_mapping[h_idx]} instead of {r_idx}"))
                    return False
            else:
                index_mapping[h_idx] = r_idx

        debug(f"match: mapped {hardware} to {requirement}")
        return True

    if isinstance(requirement, expression_node):
        if requirement.op != hardware.op:
            debug(f"mismatch: {requirement.op} is not {hardware.op}")
            return False
            
        if requirement.reduce_dim is not None and hardware.reduce_dim is not None:
            if hardware.reduce_dim in index_mapping:
                if index_mapping[hardware.reduce_dim] != requirement.reduce_dim:
                    debug(f"mismatch: reduction dim mapping between {requirement} and {hardware}")
                    return False
            else:
                index_mapping[hardware.reduce_dim] = requirement.reduce_dim
        elif requirement.reduce_dim is not None or hardware.reduce_dim is not None:
            debug(f"mismatch: reduction dim undefined for one but not the other")
            return False

        if not map_and_match(requirement.left, hardware.left,
                             name_mapping, index_mapping):
            debug(f"mismatch: left side {requirement.left} vs {hardware.left}")
            return False
        if requirement.right is not None and hardware.right is not None:
            if not map_and_match(requirement.right, hardware.right,
                                 name_mapping, index_mapping):
                debug(f"mismatch: right side {requirement.right} vs {hardware.right}")
                return False
        elif requirement.right is not None or hardware.right is not None:
            debug(f"mismatch: right side undefined for one, but not the other")
            return False

        debug(f"match: mapped {hardware} to {requirement}")
        return True


def extract_operand_pairs(req: ast_node, hw: ast_node) \
        -> list[tuple[operand_ref, operand_ref]]:
    """
    Helper to extract corresponding operand leaves from two structurally matched ASTs.
    """
    if isinstance(req, operand_ref) and isinstance(hw, operand_ref):
        return [(req, hw)]
    pairs = []
    if isinstance(req, expression_node) and isinstance(hw, expression_node):
        pairs.extend(extract_operand_pairs(req.left, hw.left))
        if req.right and hw.right:
            pairs.extend(extract_operand_pairs(req.right, hw.right))
    return pairs

def extract_deepest_operation(ast : ast_node,
                              temp_counter: int) -> \
                                      tuple[ast_node,list[ast_node]]:
    """
    Finds the deepest operation in the AST and replaces it with a MOVE AST on a temporary

    :param ast: AST to extract from
    :param temp_counter: Now many temporaries were already substituted
    :return: tuple consisting of a modified AST including the substitution and a list of new requirements
    """

    if isinstance(ast, operand_ref):
        return ast,[]

    if operation.MOVE == ast.op:

        right_node = ast.right

        # MOVE(C, T0)
        if isinstance(right_node, operand_ref):
            return ast, []

        left_is_leaf = isinstance(right_node.left, operand_ref)
        right_is_leaf = right_node.right is None or \
                isinstance(right_node.right, operand_ref)

        # MOVE(C, OP(A,B))
        if left_is_leaf and right_is_leaf:
            return ast, []

        new_right, deps = extract_deepest_operation(ast.right, temp_counter)


        return expression_node(
                  op=operation.MOVE,
                  left=ast.left,
                  right=new_right
               ), deps

    left_is_leaf = isinstance(ast.left, operand_ref)
    right_is_leaf = ast.right is None or isinstance(ast.right, operand_ref)

    if left_is_leaf and right_is_leaf:
        temp_indices = recurse_indices(ast)
        temp_ref = operand_ref(name=f"T{temp_counter}",
                               indices=tuple(temp_indices))


        new_req = expression_node(operation.MOVE, left=temp_ref, right=ast)
        
        debug(f"temporary {temp_ref} created as substitute for {ast} with {new_req} as dependency")

        return temp_ref, [new_req]

    if not right_is_leaf:
        new_right, deps = extract_deepest_operation(ast.right, temp_counter)
        return expression_node(
                op=ast.op,
                left=ast.left,
                right=new_right,
                reduce_dim=ast.reduce_dim
            ), deps

    if not left_is_leaf:
        new_left, deps = extract_deepest_operation(ast.left, temp_counter)
        return expression_node(
                op=ast.op,
                left=new_left,
                right=ast.right,
                reduce_dim=ast.reduce_dim
            ), deps

    raise RuntimeError("This shouldn't be reachable")


def transform_operand(operand : operand_ref, tf : transformation) -> operand_ref:
    """
    Returns a new operand resulting from transforming an input operand
    
    :param operand: Operand to transform
    :param tf: Transformaiton to apply
    :return: New transformed operand
    """
    new_operand = copy(operand)

    idx = list(operand.indices)

    if transformation.TRANSPOSE == tf and len(idx) == 2:
        idx = [idx[1],idx[0]]
    if transformation.SCALAR_REDUCE == tf:
        if len(idx) == 2:
            if idx[0] is not None and idx[1] is None:
                idx[0] = None
            elif idx[0] is None and idx[1] is not None:
                idx[1] = None
            else:
                pass
    if transformation.COL_REDUCE == tf:
        if len(idx) == 2:
            if idx[0] is not None:
                idx[0] = None
            else:
                pass
    if transformation.ROW_REDUCE == tf:
        if len(idx) == 2:
            if idx[1] is not None:
                idx[1] = None
            else:
                pass

    return operand_ref(name=operand.name, indices=tuple(idx))

def transform_ast(ast : ast_node,
                  trans_dict : dict[str,list[transformation]]) -> ast_node:
    """
    Returns a new AST resulting from walking the input AST and transforming all
    its operands

    :param ast: Original AST
    :param trans_dict: Map of operands to a list of transformations to apply to them
    :return: New transformed AST
    """

    if isinstance(ast, operand_ref):
        new_op = ast
        for tf in trans_dict.get(ast.name, [transformation.NONE]):
            new_op = transform_operand(new_op, tf)
        return new_op

    new_left = transform_ast(ast.left, trans_dict)
    new_right = None
    if ast.right is not None:
        new_right = transform_ast(ast.right, trans_dict)

    return expression_node(
            op=ast.op,
            left=new_left, right=new_right,
            reduce_dim=ast.reduce_dim)


def get_operands(n: ast_node) -> set[str]:
    """
    Recursively collect all operands in an AST.
    :param n: root node of the AST
    :return: set of operand names
    """
    if isinstance(n,operand_ref):
        return {n.name}

    operands = get_operands(n.left)
    if n.right is not None:
        operands |= get_operands(n.right)
    return operands

def for_each_operand(n: ast_node, 
                     func : Callable[[ast_node],Any]) -> Iterator[Any]:
    """
    Recursively iterate through all operands and apply a function to all of them,
    yielding the return values.
    :param n: root node of the AST
    :param func: function to apply to operands
    :return: Iterator through return values of the supplied
             function applied to operands of the AST
    """
    if isinstance(n,operand_ref):
        yield func(n)

    # yield is NOT return, code picks up from after the yield so need to
    # check if node is an expression explicitly
    if isinstance(n,expression_node):
        yield from for_each_operand(n.left, func)
        if n.right is not None:
            yield from for_each_operand(n.right, func)

def for_each_expression(n: ast_node, func : Callable[[ast_node],Any]) -> Iterator[Any]:
    """
    Recursively iterate through all expressions and apply a function to all of them,
    yielding the return values.
    :param n: root node of the AST
    :param func: function to apply to expressions
    :return: Iterator through return values of the supplied
             function applied to expressions of the AST
    """
    if isinstance(n,expression_node):
        yield func(n)
        
        yield from for_each_expression(n.left, func)
        if n.right is not None:
            yield from for_each_expression(n.right, func)


def get_operand_io(n: ast_node, opd : str) -> tuple[bool,bool]:
    """
    Find out if the operator is used as input,output or both in the AST
    :param n: root node of the AST
    :param opd: name of the operand to search for
    :return: first value is True if the operand is used as input, 
             second value if it is used as output
    """

    io = [False,False]

    def mark_io(n: ast_node):
        if isinstance(n, expression_node):
            if isinstance(n.left, operand_ref):
                if operation.MOVE == n.op:
                    if n.left.name == opd:
                        io[1] = True
                if n.left.name == opd:
                    io[0] = True
            if n.right is not None:
                if isinstance(n.right, operand_ref):
                    if n.right.name == opd:
                        io[0] = True

    for _ in for_each_expression(n, mark_io):
        pass

    return tuple(io)

TF_DEGENERACIES = {
    transformation.ROW_REDUCE : {transformation.SCALAR_REDUCE,
                                 transformation.NONE},
    transformation.COL_REDUCE : {transformation.SCALAR_REDUCE,
                                 transformation.NONE},
    transformation.SCALAR_REDUCE : {transformation.NONE},
    transformation.TRANSPOSE : {transformation.NONE},
}

def is_transformation_weak(opd : operand_ref,
                           tf : transformation) -> bool:
    """
    Check if a transformation is "weak" (either doesn't do anything or
    is equivalent to a "stronger" transformation)
    """
    
    if len(opd.indices) != 2:
        return True

    if tf not in TF_DEGENERACIES:
        return False
    for stf in TF_DEGENERACIES[tf]:
        if transform_operand(opd,tf) == transform_operand(opd,stf):
            return True

    return False


def has_weak_transformations(ast : ast_node, 
                             trans_dict : dict[str,list[transformation]]) -> bool:
    """
    Checks if any of the transformations in trans_dict is "weak"/redundant

    :param ast: AST to check
    :param trans_dict: Operand transformations to check
    :return: True if any weak transformations are present, False otherwise
    """

    return any(for_each_operand(
                ast,
                lambda opd : any(
                    is_transformation_weak(
                        opd, tf)
                    for tf in trans_dict[opd.name])))


def decimate_index(node : ast_node, idx : str) -> ast_node:
    
    if isinstance(node, operand_ref):

        new_indices = [i if i != idx else None for i in node.indices]
        if len(new_indices) > 2:
            pruned_indices = [new_indices[0]]
            for i in new_indices[1:-1]:
                if i is not None:
                    pruned_indices.append(i)
            pruned_indices.append(new_indices[-1])

            new_indices = pruned_indices

        return operand_ref(
                name=node.name,
                indices=tuple(new_indices)
                )
    
    l = decimate_index(node.left, idx)
    r = None
    if node.right is not None:
        r = decimate_index(node.right, idx)

    return expression_node(
            op = node.op, 
            left = l,
            right = r,
            reduce_dim= node.reduce_dim)

def generate_variants(node: ast_node) -> list[ast_node]:
    """
    Generates the original AST, ASTs where reductions are applied and
    Commutative variants with swapped operands for MUL/ADD

    :param node: original AST
    :return: list of original AST and generated variants
    """

    debug(f"Making variants of {node}")

    if isinstance(node, operand_ref):
        return [node]
        
    left_vars = generate_variants(node.left)
    right_vars = generate_variants(node.right) if node.right else [None]
    
    variants = []
    for l in left_vars:
        for r in right_vars:
            # 1. Standard structural variant
            variants.append(expression_node(node.op, l, r, node.reduce_dim))

            # 2. Commutative variants
            if r is not None and node.op in (operation.ADD, operation.MUL):
                variants.append(expression_node(node.op,
                                                left=r, right=l,
                                                reduce_dim=node.reduce_dim))

            
            # 3. If this is a reduction, ALSO append the operand of the reduction
            #    with decimated reduction dimension
            if node.op == operation.REDUCE_SUM:
                variants.append(decimate_index(l, node.reduce_dim))

    debug(f"Generated variants: {variants}")
    return variants

def validate_variance(req: ast_node, hw: ast_node,
                      index_mapping: dict[str, str]) -> bool:
    """
    Check the following condition: 
    If the math requires an operand to vary along dimension R, and the hardware 
    vectorizes along dimension H (where H maps to R),
    the hardware operand MUST contain H.
    """
    pairs = extract_operand_pairs(req, hw)
    for req_op, hw_op in pairs:
        req_shape = req_op.get_shape()
        hw_shape = hw_op.get_shape()
        
        for h_idx, r_idx in index_mapping.items():
            if r_idx in req_shape and h_idx not in hw_shape:
                debug(f"hw shape of {hw_op}: {hw_shape}")
                debug(f"Variance failure: {r_idx} in req, but {h_idx} not in hw")
                return False
    return True

@dataclass
class req_solution_step:
    """
    One step of a solution to an AST requirement
    """
    hw_ast : ast_node
    transformations : dict[str, list[transformation]]
    name_mapping : dict[str,str]
    index_mapping : dict[str,str]

def transform_and_match(req: ast_node, hw_ast: ast_node) -> Iterator[req_solution_step]:
    """
    Generator that transforms an HW AST and yields the transformed ast if
    it matches the requirement

    :param req: Mathematical requirement
    :param hw_ast: Single AST representing HW operation
    :return: solution steps containing matching transformed HW ASTs
    """
    if hw_ast.op != operation.MOVE:
        raise ValueError(
                f"HW AST root op must be MOVE, instead it's {hw_ast.op.name}")

    tf_options = list(transformation)
    all_operands = get_operands(hw_ast)
    

    for perm in itertools.product(tf_options, repeat=len(all_operands)):
        trans_dict = {op_name: [tf] for op_name, tf in zip(all_operands, perm)}

        if has_weak_transformations(hw_ast, trans_dict):
            continue

        logical_hw = transform_ast(hw_ast, trans_dict)
        simplified_hw = simplify_ast(logical_hw)
        
        # 2. SHAPE VALIDATION: Reject impossible HW physics (Fake DOTA)
        if not simplified_hw.is_dimensionally_valid():
            continue

        index_mapping = {}
        name_mapping = {}
        
        # 3. STRICT STRUCTURAL MATCHING
        if map_and_match(req, simplified_hw, name_mapping, index_mapping):

            debug(f"success: {req} and {simplified_hw} match, validating variance")
            
            # 4. VARIANCE VALIDATION: Reject fake scalars (All-None Swallows)
            if validate_variance(req, simplified_hw, index_mapping):
                yield req_solution_step(
                    hw_ast=hw_ast,
                    transformations=trans_dict,
                    name_mapping=name_mapping,
                    index_mapping=index_mapping
                )
            else:
                debug(f"variance validation failed")



def solve_requirement(req: ast_node, hw_asts: list[ast_node],
                      temp_counter=0) \
        -> list[list[dict]]:
    """
    Generates a list of hw ASTs that solve the mathematical requirement

    :param req: Mathematical requirement to solve for
    :param hw_asts: List of hardware ASTs available for the solution
    :param temp_counter: Starting index for temporary name generation 
                         (used for recursion)
    :return: list of valid instruction chains, like [[FMA],[FMUL,FADD],[FDOTA]]
    """

    all_valid_chains = []

    req_variants = generate_variants(req)

    for variant in req_variants:

        direct_matches = [
            [match]
            for hw_ast in hw_asts
            for match in transform_and_match(variant, hw_ast)
        ]
        all_valid_chains.extend(direct_matches)

        if isinstance(variant, operand_ref):
            continue

        split_ast, dependencies = extract_deepest_operation(
                variant,
                temp_counter=temp_counter)

        if not dependencies:
            continue

        dep_solutions = solve_requirement(dependencies[0], hw_asts,
                                          temp_counter=temp_counter+1)
        remainder_solutions = solve_requirement(split_ast, hw_asts,
                                                temp_counter=temp_counter+1)


        for dep_chain in dep_solutions:
            for rem_chain in remainder_solutions:

                # This must be a temporary
                output_opd = dep_chain[-1].hw_ast.left 
                dep_tfs = dep_chain[-1].transformations[output_opd.name]

                invalid_change = False
                for opd in for_each_operand(rem_chain[0].hw_ast, lambda x : x):
                    if opd.name == output_opd.name:
                        rem_tfs = rem_chain[0].transformations[opd.name]
                        if transform_ast(output_opd, {opd.name:rem_tfs}) != \
                                transform_ast(opd, {opd.name:dep_tfs}):
                            invalid_change = True
                            break


                if not invalid_change:
                    all_valid_chains.append(dep_chain+rem_chain)

    return all_valid_chains

HW_FADD_AST = expression_node(
        op=operation.MOVE,
        left=operand_ref(name="cdreg", indices=('i',None)),
        right=expression_node(
            op=operation.ADD,
            left=operand_ref(name="adreg", indices=('i', None)),
            right=operand_ref(name="bdreg", indices=('i', None)))
        )

HW_FMUL_AST = expression_node(
        op=operation.MOVE,
        left=operand_ref(name="cdreg", indices=('i',None)),
        right=expression_node(
            op=operation.MUL,
            left=operand_ref(name="adreg", indices=('i', None)),
            right=operand_ref(name="bdreg", indices=('i', None)))
        )

HW_FMA_AST = expression_node(
        op=operation.MOVE,
        left=operand_ref(name="cdreg", indices=('i', None)),
        right=expression_node(
            op=operation.ADD,
            left=operand_ref(name="cdreg", indices=('i',None)),
            right=expression_node(
                op=operation.MUL,
                left=operand_ref(name="adreg", indices=('i',None)),
                right=operand_ref(name="bdreg", indices=('i',None))
                )
            )
        )

HW_FDOTA_AST = expression_node(
        op=operation.MOVE,
        left=operand_ref(name="cdreg", indices=('i',None)),
        right=expression_node(
            op=operation.ADD,
            left=operand_ref(name="cdreg", indices=('i',None)),
            right=expression_node(
                op=operation.REDUCE_SUM,
                left=expression_node(
                    op=operation.MUL,
                    left=operand_ref(name="adreg", indices=('i',None)),
                    right=operand_ref(name="bdreg", indices=('i',None))),
                reduce_dim='i')
            )
        )

HW_FOPA_AST = expression_node(
        op=operation.MOVE,
        left=operand_ref(name="cdreg", indices=('i','j')),
        right=expression_node(
            op=operation.ADD,
            left=operand_ref(name="cdreg", indices=('i','j')),
            right=expression_node(
                op=operation.MUL,
                left=operand_ref(name="adreg", indices=('i',None)),
                right=operand_ref(name="bdreg", indices=('j',None)))
            )
        )

HW_MMA_AST = expression_node(
        op=operation.MOVE,
        left=operand_ref(name="cdreg", indices=('i', 'j')),
        right=expression_node(
            op=operation.ADD,
            left=operand_ref(name="cdreg", indices=('i','j')),
            right=expression_node(
                op=operation.REDUCE_SUM,
                left=expression_node(
                    op=operation.MUL,
                    left=operand_ref(name="adreg", indices=('i','k')),
                    right=operand_ref(name="bdreg", indices=('k','j'))),
                reduce_dim='k')
            )
        )
