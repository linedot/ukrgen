import abc
from enum import Enum,auto

from asmgen.asmblocks.noarch import asmgen,asm_data_type

from .load_register_sequence_generator import load_register_sequence_generator as lrsg

#class loader:
#    
#    @abc.abstractmethod
#    def process(self) -> str:
#        raise NotImplementedError("Attempted to call base class method")


class register_state(Enum):
    free = auto()
    written = auto()
    in_use = auto()

class scheduler:
    def __init__(self, valid_transitions : list[tuple[register_state,register_state]]):
        self.valid_transitions = valid_transitions

    @abc.abstractmethod
    def schedule(self,
                 asmqueue : list[str,dict[str,register_state]],
                 states : dict[str,register_state]) -> list[str]:
        raise NotImplementedError("Attempted to call base class method")


class simple_scheduler(scheduler):
    def schedule(self,
                 asmqueue : list[str,dict[str,register_state]],
                 states : dict[str,register_state]) -> list[str]:

        asm_sequence = []
        for asm,st in asmqueue:
            for reg,st in st.items():
                if (states[reg],st) in self.valid_transitions:
                    states[reg] = st
                else:
                    raise ValueError(f"Invalid state transition from {states[reg]} to {st}\nvalid transitions:{self.valid_transitions}")
            asm_sequence.append(asm)
        return asm_sequence

class register_state_tracker:
    def __init__(self):
        self.register_states : dict[str,state] = {}

class offset_rollover_tracker:
    def __init__(self, asmgen : asmgen):
        self.asmgen = asmgen
        self.queue = []

    @abc.abstractmethod
    def __call__(self, address_register : int, offset : int):
        raise NotImplementedError("Attempted to call abstract method")

class imm_offset_rollover_tracker(offset_rollover_tracker):
    def __init__(self, asmgen : asmgen):
        self.asmgen = asmgen
        self.queue = []

    def __call__(self, address_register : int, offset : int):
        self.queue.append((
            self.asmgen.add_greg_imm(
                self.asmgen.greg(address_register),
                offset),
            {str(self.asmgen.greg(address_register)) : register_state.written}
            ))

class greg_offset_rollover_tracker(offset_rollover_tracker):
    def __init__(self, asmgen : asmgen, offreg : int):
        self.asmgen = asmgen
        self.offreg = offreg
        self.queue = []

    def __call__(self, address_register : int, offset : int):
        if 1 != offset:
            raise ValueError("greg_offset rollover must be called with offset=1")
        self.queue.append((
            self.asmgen.add_greg_greg(
                self.asmgen.greg(address_register),
                self.asmgen.greg(address_register),
                self.asmgen.greg(self.offreg)),
            {str(self.asmgen.greg(address_register)) : register_state.written}
            ))


class vector_load_generator:
    def __init__(self,
        asmgen        : asmgen,
        lrsg          : lrsg,
        rst           : register_state_tracker,
        scheduler     : scheduler,
        ort           : offset_rollover_tracker):
        self.asmgen = asmgen
        self.lrsg = lrsg
        self.rst = rst
        self.scheduler = scheduler
        self.ort = ort

    def generate(self, 
                 count : int, 
                 dt : asm_data_type
                 ) -> list[str,dict[str,register_state]]:

        rs = register_state

        asmqueue = []

        for i in range(count):
            ldata = self.lrsg.advance()
            
            areg = ldata["address_register"]
            dreg = ldata["data_register"]
            off  = ldata["offset"]

            state_changes = {str(self.asmgen.vreg(dreg)) : rs.written}


            asmqueue.append(
                (self.asmgen.load_vector_voff(
                    self.asmgen.greg(areg),
                    off,
                    self.asmgen.vreg(dreg), dt),
                 state_changes)
            )

            if self.ort.queue:
                asmqueue.extend(self.ort.queue)
                self.ort.queue = []

        return asmqueue

#TODO: code deduplication
class scalar_load_generator:
    def __init__(self,
        asmgen        : asmgen,
        lrsg          : lrsg,
        rst           : register_state_tracker,
        scheduler     : scheduler,
        ort           : offset_rollover_tracker):
        self.asmgen = asmgen
        self.lrsg = lrsg
        self.rst = rst
        self.scheduler = scheduler
        self.ort = ort

    def generate(self, 
                 count : int, 
                 dt : asm_data_type
                 ) -> list[str,dict[str,register_state]]:

        rs = register_state

        asmqueue = []

        for i in range(count):
            ldata = self.lrsg.advance()
            
            areg = ldata["address_register"]
            dreg = ldata["data_register"]
            off  = ldata["offset"]

            # 1/2 change
            state_changes = {str(self.asmgen.freg(dreg)) : rs.written}

            asmqueue.append(
                (self.asmgen.load_scalar_immoff(
                    self.asmgen.greg(areg),
                    off,
                    self.asmgen.freg(dreg), dt),
                 state_changes)
            )

            if self.ort.queue:
                asmqueue.extend(self.ort.queue)
                self.ort.queue = []

        return asmqueue
