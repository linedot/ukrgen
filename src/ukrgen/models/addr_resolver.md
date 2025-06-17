# Idea:

Address registers contain addresses
For each addr register save the offset from the starting address

for an operation that accesses memory (load/store) at a specified
offset to starting address, determine the address register to use

Address register can be used if the required offset can be reached
by using the register with an immediate/vector offset. 

immediate/vector offsets determined by offset ranges passed to the
constructor

Address registers can be advanced by a parameter determined amount,
resulting in an lsc_addr_add operation (decide whether the resolver
or the model should issue it)

The way registers are switched/advanced should be controlled by parameters
in some ways

# Examples:
## `addr_regs = [0,1]` `addr_offsets =[0,2]` `offset_range = [0,3]`

sequence:

```
   load toff=0
   load 1
   load 2
   load 3
   load 4
   load 5
   load 6
   load 7
   load 8
   load 9
```
## add addresses when `off=min(2, offset_required(regs,toff))`

 Transform into:
```
   load reg0+0
   load reg0+1
   load reg1+0
   load reg1+1
   reg0 <- reg0+4  (reg0=4)
   load reg0+0
   load reg0+1
   reg1 <- reg1+4  (reg1=6)
   load reg1+0
   load reg1+1
   reg0 <- reg0+4  (reg0=8)
   load reg0+0
   load reg0+1
```
Preparation for next loop iter:
```
  reg0 <- reg0+2  (reg0=10)
  reg1 <- reg1+6  (reg1=12)
```

## add addresses when `off=max(2,offset_required(regs,toff))`

Transform into:
```
  load reg0+0
  load reg0+1
  load reg0+2
  reg1 <- reg1+1  (reg1=3)
  load reg1+0
  load reg1+1
  load reg1+2
  reg0 <- reg0+6  (reg0=6)
  load reg0+0
  load reg0+1
  load reg0+2
  reg1 <- reg1+6  (reg1=9)
  load reg1+0
```

Preparation for next loop iter:
```
  reg0 <- reg0+4  (reg0=10)
  reg1 <- reg1+3  (reg1=12)
```
