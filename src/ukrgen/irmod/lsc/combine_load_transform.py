"""
Combining loads and transform instructions.

This is relevant for ISAs that can use memory locations as operands.
Example: AVX FMAs, transform
```
vmovupd 64(%rax), %zmm0
vfmadd213pd %zmm1, %zmm0, %zmm2
```
into
```
vfmadd213pd %zmm1, 64(%rax), %zmm2
```
"""

# Notes:
#   - Need to know which operand can be memory
#   - Prepare for ISAs where an operand MUST be memory (NVIDIA tcgen05...)
#   - Needs to know which operand is which component (should be known?)
#   - Offsets/Addressing might be different than for loads
