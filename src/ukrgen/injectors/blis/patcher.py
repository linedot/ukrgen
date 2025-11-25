import logging
import shutil
import pathlib
import os

from typing import Callable
from copy import deepcopy

from mako.template import Template, exceptions
from patch_ng import fromstring


from asmgen.registers import asm_data_type as adt

from .templates import (
    components as tpl_components,
    files as tpl_files,
    patches as tpl_patches,
    kernels as tpl_kernels
)

from ...composers.stage_engine import stage_engine
from ...composers.gemm import gemm_context

from ...composers.stages import (
    composition_stage,
    support_stage,
    datatype_stage,
    dimension_stage,
    mm_tif_stage,
    lsc_model_stage,
    lsc_mru_stage,
    lsc_schedule_stage,
    specialize_lsc_stage,
    blis_ukr_codegen_stage
)

class config_prolog:
    def __init__(self, config : dict[str,str|list[str]]):
        self.config = config

        self.param_callbacks : dict[str,
                                    Callable[[gemm_context],
                                             str|list[str]]] = dict()

    def add_param_callback(self, name : str,
                           cb : Callable[[gemm_context],str|list[str]]):
        self.param_callbacks[name] = cb

    def __call__(self, stage: composition_stage):
        params = stage.get_parameter_names()

        if not params:
            return

        for pname in params:
            if pname in self.param_callbacks:
                stage.set_param(pname, 
                    self.param_callbacks[pname](stage.context))
            elif pname in self.config:
                stage.set_param(pname, self.config[pname])
            else:
                stage.set_param(pname, stage.params[pname].default)

class blis_kernel:
    def __init__(self,
                 csctx : gemm_context,
                 rsctx : gemm_context,
                 csrsctx : gemm_context):
        self.cs = csctx
        self.rs = rsctx
        self.csrs = csrsctx

class blis_patcher:
    def __init__(self):

        self.tpl_components = deepcopy(tpl_components)
        self.tpl_files = deepcopy(tpl_files)
        self.tpl_patches = deepcopy(tpl_patches)
        self.tpl_kernels = deepcopy(tpl_kernels)
        
        self.extra_params : dict[str,str] = dict()
        self.kernels : list[blis_kernel] = []

    def make_kernels(self,
                     configurations : list[dict[str,str|list[str]]],
                     params : dict[str,str|list[str]]):
        stages = [
            support_stage,
            datatype_stage,
            dimension_stage,
            mm_tif_stage,
            lsc_model_stage,
            specialize_lsc_stage,
            lsc_mru_stage,
            lsc_schedule_stage,
            blis_ukr_codegen_stage]

        stride_configs = [
            ([],["C"],"_cs"),
            (["C"],[],"_rs"),
            (["C"],["C"],"_csrs"),
        ]

        def get_fnname(ctx : gemm_context):
            if ctx.component_dts["C"] == adt.FP32:
                dtchar = "s"
            elif ctx.component_dts["C"] == adt.FP64:
                dtchar = "d"


            m = ctx.params["m"].value
            n = ctx.params["n"].value
            k = ctx.params["k"].value

            cstrides = ctx.strides["C"]

            suf = "_"
            if cstrides[1] is not None:
                suf += "cs"
            if cstrides[0] is not None:
                suf += "rs"

            
            return f"bli_{dtchar}gemm_{params['configname']}_{m}Vx{n}x{k}{suf}"



        for config in configurations:
            contexts = []
            for rs, cs, suf in stride_configs:
                #CS
                config["row-strides"] = rs
                config["column-strides"] =cs

                ctx = gemm_context()
                prolog = config_prolog(config)
                prolog.add_param_callback("function-name", get_fnname)

                se = stage_engine(stages=stages,
                                  ctx=ctx,
                                  prolog=prolog)

                se.run()


                contexts.append(ctx)


            self.kernels.append(
                blis_kernel(csctx=contexts[0],
                            rsctx=contexts[1],
                            csrsctx=contexts[2]))



    def prepare_code(self, params : dict[str,str] = dict()):
        to_parse : set[str] = set()


        param_kernels = []

        ktpl = deepcopy(self.tpl_kernels["gemm"])

        for kernel in self.kernels:

            # general parameters will be the same across all three, so
            # just use cs ctx
            ctx = kernel.cs

            assert ctx.component_dts["C"] == ctx.component_dts["A"]
            assert ctx.component_dts["C"] in [adt.FP32,adt.FP64]

            if ctx.component_dts["C"] == adt.FP32:
                dtchar = "s"
                param_type = "FLOAT"
            elif ctx.component_dts["C"] == adt.FP64:
                dtchar = "d"
                param_type = "DOUBLE"

            m = ctx.params["m"].value
            n = ctx.params["n"].value
            k = ctx.params["k"].value

            params[f"nr_{dtchar}"] = n
            params[f"kunroll_{dtchar}"] = k

            # remove the _cs
            ukr_name = ctx.params["function-name"].value[:-3]

            ksrc = Template(ktpl).render(
                    configname=params["configname"],
                    blis_ukr_name=ukr_name,
                    ukr="gemm",
                    dtchar=dtchar,
                    ukr_extra_def = ctx.gen.c_simd_size_function,
                    kunroll=k,
                    mr=f"{m}*get_simd_size()/sizeof({param_type.lower()})",
                    nr=n
                    )

            self.tpl_files[f"kernels/${{configname}}/3/{ukr_name}_cs.s"] = \
                    kernel.cs.asmblocks["full_function"]
            self.tpl_files[f"kernels/${{configname}}/3/{ukr_name}_rs.s"] = \
                    kernel.rs.asmblocks["full_function"]
            self.tpl_files[f"kernels/${{configname}}/3/{ukr_name}_csrs.s"] = \
                    kernel.csrs.asmblocks["full_function"]
            self.tpl_files[f"kernels/${{configname}}/3/{ukr_name}.c"] = ksrc

            param_kernels.append(
                    ("GEMM",param_type,ukr_name)
                    )

        params["cntx_extra_defs"] = self.kernels[0].cs.gen.c_simd_size_function

        params["kernels"] = param_kernels

        for tpl in [self.tpl_components,self.tpl_files,self.tpl_patches]:

            to_resolve = {k for k in tpl.keys() if k not in params}

            resolved_keys : set[str] = set()
            resolved_vals : set[str] = set()
            while to_resolve:
                kr = 0
                vr = 0
                keys = deepcopy(to_resolve)
                for key in keys:
                    #Key
                    try:
                        key_r = Template(key).render(**params)
                        if key not in resolved_keys:
                            kr += 1
                        resolved_keys.add(key)
                    except:
                        pass
                    #Val
                    try:
                        val_r = Template(tpl[key]).render(**params)
                        if key not in resolved_vals:
                            vr += 1
                        resolved_vals.add(key)
                    except:
                        pass
                    if key in resolved_vals:
                        tpl[key] = val_r
                    if key in resolved_keys:
                        tpl[key_r] = tpl[key]
                        if key_r != key:
                            del tpl[key]
                    if (key in resolved_keys) and not (key in resolved_vals):
                        to_resolve.remove(key)
                        to_resolve.add(key_r)
                    if (key in resolved_keys) and (key in resolved_vals):
                        params[key_r] = tpl[key_r]
                        to_resolve.remove(key)
                        resolved_keys.remove(key)
                        resolved_vals.remove(key)

                if (vr == 0) and (kr == 0):
                    print("Still to resolve")
                    for c in to_resolve:
                        print(f"===========================")
                        print(f"{c}:")
                        try:
                            Template(key).render(**params)
                        except:
                            print(exceptions.text_error_template().render())
                        try:
                            Template(tpl[key]).render(**params)
                        except:
                            print(exceptions.text_error_template().render())
                    raise RuntimeError("Nothing new resolved")


    def patch(self, blis_dir : str, out_dir : str):

        log = logging.getLogger("BLISPATCH")
        log.setLevel(logging.DEBUG)
        
        shutil.copytree(blis_dir, out_dir)

        for key,pdata in self.tpl_patches.items():

            print(f"patching {key}")            

            bpatch = fromstring(pdata.encode("UTF-8"))

            if not bpatch:
                print(f"Error parsing patch:\n{pdata}")

            bpatch.apply(root=out_dir)

        for key,fdata in self.tpl_files.items():

            filepath = os.path.join(out_dir,key)
            filedir = os.path.dirname(filepath)
            if not os.path.isdir(filedir):
                pathlib.Path(filedir).mkdir(parents=True, exist_ok=True)
            if os.path.isfile(filepath):
                raise RuntimeError("Can't copy generated file {key} - file already exists")

            with open(filepath,"w") as f:
                f.write(fdata)
