from ukrgen.composers.stages.composition import composition_stage

def inject_params(stage : composition_stage,
                  params : dict[str,str|list[str]]):

    for pname in stage.get_parameter_names():
        pval = stage.get_param(pname).default
        if pname in params:
            pval = params[pname]

        stage.set_param(pname,pval)
