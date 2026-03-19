from ukrgen.flow.stages.composition import stage

def inject_params(s : stage,
                  params : dict[str,str|list[str]]):

    for pname in s.get_parameter_names():
        pval = s.get_param(pname).default
        if pname in params:
            pval = params[pname]

        s.set_param(pname,pval)
