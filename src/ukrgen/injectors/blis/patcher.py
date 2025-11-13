import logging
import shutil
import pathlib
import os

from copy import deepcopy

from mako.template import Template
from patch_ng import fromstring

from .templates import (
    components as tpl_components,
    files as tpl_files,
    patches as tpl_patches
)

class blis_patcher:
    def __init__(self):

        self.tpl_components = deepcopy(tpl_components)
        self.tpl_files = deepcopy(tpl_files)
        self.tpl_patches = deepcopy(tpl_patches)
        pass

    def make_kernels(self, configuration):
        pass

    def prepare_code(self, params : dict[str,str] = dict()):
        to_parse : set[str] = set()

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
                    #print("Still to resolve")
                    #for c in to_resolve:
                    #    print(f"===========================")
                    #    print(f"{c}:")
                    #    try:
                    #        Template(key).render(**params)
                    #    except:
                    #        print(exceptions.text_error_template().render())
                    #    try:
                    #        Template(tpl[key]).render(**params)
                    #    except:
                    #        print(exceptions.text_error_template().render())
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
