import abc

from ..sequence_generator import sequence_generator

class load_register_sequence_generator(sequence_generator):
    required_parameters = [
        "address_register_list",
        "address_register_list_start",
        "offset_start_list",
        "offset_threshold_list",
        "offset_reset_value_list",
        "offset_step",
        "max_offset",
        "data_register_list",
        "data_register_list_start"
    ]
    side_effects = [
        "on_offset_rollover"
    ]

    def initialize(self):
        self.address_register_list = self.internal_parameters["address_register_list"]
        if not (len(self.address_register_list) == len(set(self.address_register_list))):
            raise ValueError("non-unique-entries in address register list")
        self.address_register_list_position = self.internal_parameters["address_register_list_start"]


        self.max_offset = self.internal_parameters["max_offset"]

        self.offsets = {reg : offset  for (reg,offset) in \
            zip(self.address_register_list,
                self.internal_parameters["offset_start_list"])}

        self.offset_thresholds = {reg : offset  for (reg,offset) in \
            zip(self.address_register_list,
                self.internal_parameters["offset_threshold_list"])}

        self.offset_reset_value_list = {reg : offset  for (reg,offset) in \
            zip(self.address_register_list,
                self.internal_parameters["offset_reset_value_list"])}

        for reg in self.address_register_list:
            offset = self.offsets[reg]
            threshold = self.offset_thresholds[reg]
            reset_value = self.offset_reset_value_list[reg]
            if not ( offset <= threshold):
                raise ValueError(f"starting offset ({offset}) for reg {reg} higher than it's threshold ({threshold})")
            if not ( offset <= self.max_offset):
                raise ValueError(f"starting offset ({offset}) for reg {reg} higher than max offset ({self.max_offset})")
            if not ( reset_value <= threshold):
                raise ValueError(f"offset reset value ({reset_value}) for reg {reg} higher than it's threshold ({threshold})")
            if not ( reset_value <= self.max_offset):
                raise ValueError(f"offset reset value ({reset_value}) for reg {reg} higher than max offset ({self.max_offset})")

        self.offset_step = self.internal_parameters["offset_step"]


        self.data_register_list = self.internal_parameters["data_register_list"]
        if not (len(self.data_register_list) == len(set(self.data_register_list))):
            raise ValueError("non-unique-entries in data register list")
        self.data_register_list_position = self.internal_parameters["data_register_list_start"]

        self.on_offset_rollover = self.side_effect_handlers["on_offset_rollover"]

    @abc.abstractmethod
    def advance(self):
        raise NotImplementedError("Tried to call abstract method")


class lrsg_simple(load_register_sequence_generator):

    def advance(self) -> dict[str,int]: 

        areg = self.address_register_list[self.address_register_list_position]
        dreg = self.data_register_list[self.data_register_list_position]

        current_values = {
            "address_register" : areg,
            "data_register"    : dreg,
            "offset" : self.offsets[areg],
        }


        self.offsets[areg] += self.offset_step
        if (self.offsets[areg] > self.max_offset) or (self.offsets[areg] > self.offset_thresholds[areg]):
            increase_by = self.offsets[areg]-self.offset_reset_value_list[areg]
            self.on_offset_rollover(address_register=areg, offset=increase_by)
            self.offsets[areg] = self.offset_reset_value_list[areg]
            self.address_register_list_position = (self.address_register_list_position+1) % len(self.address_register_list)

        self.data_register_list_position = (self.data_register_list_position+1) % len(self.data_register_list)

        return current_values
