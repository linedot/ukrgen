import logging
import shutil
import pathlib
import os

from typing import Callable
from copy import deepcopy

from mako.template import Template, exceptions

from asmgen.registers import asm_data_type as adt

from .templates import (
    components as tpl_components,
    files as tpl_files,
    patches as tpl_patches,
    kernels as tpl_kernels
)

from ...flow.stage_engine import stage_engine
from ...flow.ukr_context import ukr_context

from ...flow.stages import (
    stage,
    support_stage,
    datatype_stage,
    dimension_stage,
    mm_sto_stage,
    lsc_model_stage,
    irmod_inserter_stage,
    lsc_mru_stage,
    lsc_schedule_stage,
    specialize_lsc_stage,
    blis_ukr_codegen_stage
)

class config_prolog:
    def __init__(self, config : dict[str,str|list[str]]):
        self.config = config

        self.param_callbacks : dict[str,
                                    Callable[[ukr_context],
                                             str|list[str]]] = dict()

    def add_param_callback(self, name : str,
                           cb : Callable[[ukr_context],str|list[str]]):
        self.param_callbacks[name] = cb

    def __call__(self, s: stage):
        params = s.get_parameter_names()

        if not params:
            return

        for pname in params:
            if pname in self.param_callbacks:
                s.set_param(pname, 
                    self.param_callbacks[pname](s.context))
            elif pname in self.config:
                s.set_param(pname, self.config[pname])
            else:
                s.set_param(pname, s.params[pname].default)

class blis_kernel:
    def __init__(self,
                 nsctx : ukr_context,
                 csctx : ukr_context,
                 rsctx : ukr_context,
                 csrsctx : ukr_context):
        self.ns = nsctx
        self.cs = csctx
        self.rs = rsctx
        self.csrs = csrsctx

class blis_patcher:
    def __init__(self, parent_configname : str = None):

        self.tpl_components = deepcopy(tpl_components)
        self.tpl_files = deepcopy(tpl_files)
        self.tpl_patches = deepcopy(tpl_patches)
        self.tpl_kernels = deepcopy(tpl_kernels)
        
        self.extra_params : dict[str,str] = dict()
        self.kernels : list[blis_kernel] = []
        self.existing_kernels : list[tuple[str,str,bool]] = []

        self.parent_configname = parent_configname


    def add_existing_kernel(self,
                            blis_ukr_type : str,
                            blis_data_type : str,
                            function_name : str):
        self.existing_kernels.append((
            blis_ukr_type, blis_data_type, function_name
            ))

    def make_kernels(self,
                     configurations : list[dict[str,str|list[str]]],
                     params : dict[str,str|list[str]]):
        stages = [
            support_stage,
            datatype_stage,
            dimension_stage,
            mm_sto_stage,
            lsc_model_stage,
            specialize_lsc_stage,
            irmod_inserter_stage,
            lsc_mru_stage,
            lsc_schedule_stage,
            blis_ukr_codegen_stage]

        stride_configs = [
            ([],[],"_nostride"),
            ([],["C"],"_cs"),
            (["C"],[],"_rs"),
            (["C"],["C"],"_csrs"),
        ]

        def get_fnname(ctx : ukr_context):
            if ctx.component_dts["C"] == adt.FP32:
                dtchar = "s"
            elif ctx.component_dts["C"] == adt.FP64:
                dtchar = "d"


            m = ctx.params["m"].value
            n = ctx.params["n"].value
            k = ctx.params["k"].value

            cstrides = ctx.strides["C"]
            
            vecdir = ctx.params["vecdir"].value
            mr = f"{m}" + ("V" if "M" == vecdir else "")
            nr = f"{n}" + ("V" if "N" == vecdir else "")

            return f"bli_{dtchar}gemm_{params['configname']}_{mr}x{nr}x{k}"



        for config in configurations:
            contexts = []
            for rs, cs, suf in stride_configs:
                #CS
                config["row-strides"] = rs
                config["column-strides"] =cs

                ctx = ukr_context()
                prolog = config_prolog(config)
                prolog.add_param_callback("function-name", get_fnname)

                se = stage_engine(stages=stages,
                                  ctx=ctx,
                                  prolog=prolog)

                se.run()


                contexts.append(ctx)


            self.kernels.append(
                blis_kernel(nsctx=contexts[0],
                            csctx=contexts[1],
                            rsctx=contexts[2],
                            csrsctx=contexts[3]))
                            #rsctx=None,
                            #csrsctx=contexts[2]))



    def prepare_code(self, params : dict[str,str] = dict()):
        to_parse : set[str] = set()


        param_kernels = []

        ktpl = deepcopy(self.tpl_kernels["gemm"])

        for kernel in self.kernels:

            # general parameters will be the same across all three, so
            # just use cs ctx
            ctx = kernel.cs

            # Same data type and only floats and doubles for BLIS
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

            params[f"kunroll_{dtchar}"] = k

            vecdir = ctx.params["vecdir"].value
            mr = ""
            nr = ""
            # TODO: NON-FMA
            if ctx.sup.a_tile.is_tile or ctx.sup.b_tile.is_tile or ctx.sup.c_tile.is_tile:
                raise NotImplementedError("Missing mr/nr calc for tile regs")
            if ctx.sup.c_tile.is_scalar:
                raise NotImplementedError("Missing mr/nr calc for scalar C tile")

            if "M" == vecdir:
                mr=f"{m}*get_simd_size()/sizeof({param_type.lower()})"
                nr=n
            elif "N" == vecdir:
                mr=m
                nr=f"{n}*get_simd_size()/sizeof({param_type.lower()})"

            params[f"mr_{dtchar}"] = mr
            params[f"nr_{dtchar}"] = nr

            # remove the _cs
            ukr_name = ctx.params["function-name"].value

            ksrc = Template(ktpl).render(
                    configname=params["configname"],
                    blis_ukr_name=ukr_name,
                    ukr="gemm",
                    dtchar=dtchar,
                    ukr_extra_def = ctx.gen.c_simd_size_function,
                    kunroll=k,
                    mr=mr,
                    nr=nr,
                    vecdir=vecdir
                    )

            self.tpl_files[f"kernels/${{configname}}/3/{ukr_name}_nostride.s"] = \
                    kernel.ns.asmblocks["full_function"]
            if "M" == vecdir:
                self.tpl_files[f"kernels/${{configname}}/3/{ukr_name}_cs.s"] = \
                        kernel.cs.asmblocks["full_function"]
            if "N" == vecdir:
                self.tpl_files[f"kernels/${{configname}}/3/{ukr_name}_rs.s"] = \
                        kernel.rs.asmblocks["full_function"]
            self.tpl_files[f"kernels/${{configname}}/3/{ukr_name}_csrs.s"] = \
                    kernel.csrs.asmblocks["full_function"]
            self.tpl_files[f"kernels/${{configname}}/3/{ukr_name}.c"] = ksrc

            param_kernels.append(
                    ("GEMM_UKR",param_type,ukr_name)
                    )

        params["cntx_extra_defs"] = self.kernels[0].cs.gen.c_simd_size_function

        params["kernels"] = param_kernels

        params["existing_kernels"] = self.existing_kernels
        params["parent_configname"] = self.parent_configname


        for tpl in [self.tpl_components,self.tpl_files]:

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
                            Template(c).render(**params)
                        except:
                            print(exceptions.text_error_template().render())
                        try:
                            Template(tpl[c]).render(**params)
                        except:
                            print(exceptions.text_error_template().render())
                    raise RuntimeError("Nothing new resolved")

        rendered_patches = {}

        for file_key, patch_list in self.tpl_patches.items():
            try:
                rendered_file_key = Template(file_key).render(**params)
            except Exception as e:
                raise RuntimeError(f"Failed to render patch file key '{file_key}': {e}")

            rendered_patch_list = []
            for anchor, content_template, insert_after in patch_list:
                try:
                    content = Template(content_template).render(**params)
                    rendered_patch_list.append(
                            (anchor, content, insert_after))
                except Exception as e:
                    raise RuntimeError(f"Failed to render patch content for '{file_key}': {e}")

            rendered_patches[rendered_file_key] = rendered_patch_list

        self.tpl_patches = rendered_patches



    def patch(self, blis_dir : str, out_dir : str, overwrite : bool = False):

        log = logging.getLogger("BLISPATCH")

        if os.path.isdir(out_dir) and overwrite:
            shutil.rmtree(out_dir)
        
        shutil.copytree(blis_dir, out_dir)


        for rel_filepath, patch_list in self.tpl_patches.items():
            filepath = os.path.join(out_dir, rel_filepath)
            
            if not os.path.exists(filepath):
                raise RuntimeError(f"File not found: {filepath}")
                
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            modified = False
            
            for anchor, insert_text, insert_after in patch_list:
                    
                idx = content.find(anchor)
                if idx != -1:
                    if insert_after:
                        content = content[:idx+len(anchor)] + \
                                insert_text + \
                                content[idx+len(anchor):]
                    else:
                        # Slice the string and insert the new text before the anchor
                        content = content[:idx] + insert_text + content[idx:]
                    modified = True
                else:
                    log.debug(f"Error: Anchor not found in {filepath}")
                    log.debug(("--- Anchor text expected ---\n"
                              f"{anchor}"
                              "\n----------------------------"))
                    raise RuntimeError("Anchor not found in file to patch")

            # Only write back to the disk if we successfully applied a patch
            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                log.debug(f"Successfully patched: {filepath}")


        for key,fdata in self.tpl_files.items():

            filepath = os.path.join(out_dir,key)
            filedir = os.path.dirname(filepath)
            if not os.path.isdir(filedir):
                pathlib.Path(filedir).mkdir(parents=True, exist_ok=True)
            if os.path.isfile(filepath):
                raise RuntimeError("Can't copy generated file {key} - file already exists")

            with open(filepath,"w") as f:
                f.write(fdata)
