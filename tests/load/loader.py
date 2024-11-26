import unittest

from algobuild.load import (
        vector_load_generator,
        lrsg_simple,
        register_state_tracker,
        register_state as rs,
        simple_scheduler,
        greg_offset_rollover_tracker)

from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.noarch import asm_data_type as adt

class test_vector_load_generator(unittest.TestCase):
    def test_simple_load_sequence(self):
        asmgen = rvv()


        data_registers = [0,2,4,6]
        address_registers = [0, 1]
        vlen_reg = 2
        rollovers = []
        parameters = {
            "address_register_list" : address_registers,
            "address_register_list_start" : 0,
            "offset_start_list" : [0, 0],
            "offset_threshold_list" : [0, 0],
            "offset_reset_value_list" : [0, 0],
            "offset_step" : 1,
            "max_offset" : 0,
            "data_register_list" : data_registers,
            "data_register_list_start" : 0
        }
        gort = greg_offset_rollover_tracker(asmgen, vlen_reg)

        side_effect_handlers = {
            "on_offset_rollover" : gort
        }
        lrsg = lrsg_simple(parameters=parameters,
                           side_effect_handlers=side_effect_handlers)

        rst = register_state_tracker()

        sgr = lambda x : str(asmgen.greg(x))
        svr = lambda x : str(asmgen.vreg(x))
        initial_states = {svr(r): rs.free for r in data_registers} | \
            {sgr(r): rs.free for r in address_registers}

        scheduler = simple_scheduler(
            valid_transitions=[
                (rs.free,rs.free),
                (rs.free,rs.written),
                (rs.written,rs.written),
                (rs.written,rs.free),
                ])


        vlg = vector_load_generator(
            asmgen=asmgen,
            lrsg=lrsg,
            rst=rst,
            scheduler=scheduler,
            ort=gort)

        asm_queue = vlg.generate(count=8, dt=adt.FP64)

        asm_sequence = scheduler.schedule(asm_queue, initial_states)

        expected_sequence= [
            "\"vle64.v v0, (t0)\\n\\t\"\n",
            "\"add t0,t0,t2\\n\\t\"\n",
            "\"vle64.v v2, (t1)\\n\\t\"\n",
            "\"add t1,t1,t2\\n\\t\"\n",
            "\"vle64.v v4, (t0)\\n\\t\"\n",
            "\"add t0,t0,t2\\n\\t\"\n",
            "\"vle64.v v6, (t1)\\n\\t\"\n",
            "\"add t1,t1,t2\\n\\t\"\n",
            "\"vle64.v v0, (t0)\\n\\t\"\n",
            "\"add t0,t0,t2\\n\\t\"\n",
            "\"vle64.v v2, (t1)\\n\\t\"\n",
            "\"add t1,t1,t2\\n\\t\"\n",
            "\"vle64.v v4, (t0)\\n\\t\"\n",
            "\"add t0,t0,t2\\n\\t\"\n",
            "\"vle64.v v6, (t1)\\n\\t\"\n",
            "\"add t1,t1,t2\\n\\t\"\n"
        ]
        self.assertEqual(expected_sequence, asm_sequence)
