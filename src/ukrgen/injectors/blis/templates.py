components = {
    "license" : """
/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) ${year}, ${author}

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
""",
    "cntx_set_kernels" : """
    bli_cntx_set_ukrs(
        cntx,
    // inherited kernels
    % for kind,type,name in existing_kernels:
        BLIS_${kind}, BLIS_${type}, ${name},
    % endfor

    // generated kernels
    % for kind,type,name in kernels:
        BLIS_${kind}, BLIS_${type}, ${name},
    % endfor

        BLIS_VA_END
    );
""",
    "cntx_set_kernel_prefs" : """
    bli_cntx_set_ukr_prefs(
        cntx,

    % for kind,type,_ in kernels:
        BLIS_${kind}_ROW_PREF, BLIS_${type}, FALSE,
    % endfor

        BLIS_VA_END
    );
""",
    "cntx_set_blocksizes" : """

    const uint32_t mr_in_mc_d = bli_env_get_var("BLIS_MR_IN_MC_D", 16);
    const uint32_t mr_in_mc_s = bli_env_get_var("BLIS_MR_IN_MC_S", 16);
    const uint32_t nr_in_nc_d = bli_env_get_var("BLIS_NR_IN_NC_D", 200);
    const uint32_t nr_in_nc_s = bli_env_get_var("BLIS_NR_IN_NC_S", 200);
    const uint32_t kc_d = bli_env_get_var("BLIS_KC_D", 256);
    const uint32_t kc_s = bli_env_get_var("BLIS_KC_S", 256);

    // TODO: vecdir
    const uint32_t mr_d = ${mr_d};
    const uint32_t mr_s = ${mr_s};
    const uint32_t nr_d = ${nr_d};
    const uint32_t nr_s = ${nr_s};

    const uint32_t nc_d = nr_in_nc_d*nr_d;
    const uint32_t nc_s = nr_in_nc_s*nr_s;

    const uint32_t mc_d = mr_in_mc_d*mr_d;
    const uint32_t mc_s = mr_in_mc_s*mr_s;

    // Initialize level-3 blocksize objects with architecture-specific values.
    //                                              s        d        c        z
    bli_blksz_init_easy( &blkszs[ BLIS_MR ],     mr_s,    mr_d,      -1,      -1 );
    bli_blksz_init_easy( &blkszs[ BLIS_NR ],     nr_s,    nr_d,      -1,      -1 );
    bli_blksz_init_easy( &blkszs[ BLIS_MC ],     mc_s,    mc_d,      -1,      -1 );
    bli_blksz_init_easy( &blkszs[ BLIS_KC ],     kc_s,    kc_d,      -1,      -1 );
    bli_blksz_init_easy( &blkszs[ BLIS_NC ],     nc_s,    nc_d,      -1,      -1 );

    bli_cntx_set_blkszs
    (
      cntx,

      // level-3
      BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
      BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
      BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
      BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
      BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,

      BLIS_VA_END
    );
"""
}

files = {
    "config/${configname}/bli_kernel_defs_${configname}.h" : """
${license}
// Nothing here atm
""",
    "kernels/${configname}/bli_family_${configname}.h" : """
${license}
// Nothing here atm
""",
    "kernels/${configname}/bli_kernels_${configname}.h" : """
${license}
<% dtchars = {"FLOAT" : "s", "DOUBLE" : "d"} %>
% for kind,type,name in kernels:
GEMM_UKR_PROT(${type.lower()}, ${dtchars[type]}, ${name[5:]})
% endfor
""",
    "config/${configname}/bli_cntx_init_${configname}.c" : """

${license}

#include "blis.h"

${cntx_extra_defs}

void bli_cntx_init_${configname}( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_${configname}_ref( cntx );

	// -------------------------------------------------------------------------

    ${cntx_init_extra}


    ${cntx_set_kernels}

    ${cntx_set_kernel_prefs}

    ${cntx_set_blocksizes}
}
""",
    "config/${configname}/make_defs.mk" : """
THIS_CONFIG    := ${configname}

ifeq ($(CC),)
CC             := gcc
CC_VENDOR      := gcc
endif

CMISCFLAGS     := ${archflags}
CPICFLAGS      := -fPIC
CWARNFLAGS     := -Wall -Wno-unused-function -Wfatal-errors

ifneq ($(DEBUG_TYPE),off)
CDBGFLAGS      := -g
endif

ifeq ($(DEBUG_TYPE),noopt)
COPTFLAGS      := -O0
else
COPTFLAGS      := -O3
endif

CKOPTFLAGS     := $(COPTFLAGS)

ifeq ($(CC_VENDOR),gcc)
CVECFLAGS      := ${archflags} -O3 -ffast-math
else
ifeq ($(CC_VENDOR),clang)
CVECFLAGS      := ${archflags} -funsafe-math-optimizations -ffp-contract=fast
else
$(error gcc or clang is required for this configuration.)
endif
endif

# Store all of the variables here to new variables containing the
# configuration name.
$(eval $(call store-make-defs,$(THIS_CONFIG)))
"""
}

# filename : (anchor_lines, template, insert_after)
patches = {
    "config_registry" : [
        ("# Generic architectures.\ngeneric:     generic",
         """
# generated
% if parent_configname is not None:
${configname}: ${configname}/${configname}/${parent_configname}
% else:
${configname}: ${configname}
% endif

         """, False)
        ],
    "frame/base/bli_arch.c" : [
        ("		// Generic microarchitecture.\n		#ifdef BLIS_FAMILY_GENERIC",
         """
		// Generated microarchitecture.
		#ifdef BLIS_FAMILY_${configname.upper()}
		id = BLIS_ARCH_${configname.upper()};
		#endif

""", False),
        ("    \"generic\"\n};",
         """
    "${configname}",

""", False)
        ],
    "frame/include/bli_arch_config.h" : [
        ("// -- Generic --\n\n#ifdef BLIS_FAMILY_GENERIC",
         """
// -- ukrgen --

#ifdef BLIS_FAMILY_${configname.upper()}
#include "bli_family_${configname}.h"
#endif

""", False),
        ("#include \"bli_kernels_sifive_x280.h\"\n#endif",
         """
#ifdef BLIS_KERNELS_${configname.upper()}
#include "bli_kernels_${configname}.h"
#endif

""", True)
        ],
    "frame/include/bli_gentconf_macro_defs.h" : [
        ("// -- Generic architectures ----------------------------------------------------",
        """
// -- Generated architectures --------------------------------------------------

#ifdef BLIS_CONFIG_${configname.upper()}
#define INSERT_GENTCONF_${configname.upper()} GENTCONF( ${configname.upper()}, ${configname} )
#else
#define INSERT_GENTCONF_${configname.upper()}
#endif
""", False),
        ("INSERT_GENTCONF_SIFIVE_RVV \\\nINSERT_GENTCONF_SIFIVE_X280 \\",
         """
INSERT_GENTCONF_${configname.upper()} ${"\\\\"}
${"\\\\"}""", True)
        ],
    "frame/include/bli_type_defs.h" : [
        ("	// Generic architecture/configuration\n	BLIS_ARCH_GENERIC,",
        """
	// Generated architecture/configuration
	BLIS_ARCH_${configname.upper()},
""", False)
        ],
    }


kernels = {
    "gemm" : """
#include "blis.h"

${ukr_extra_def}

extern void ${blis_ukr_name}_nostride(dim_t m, dim_t n, dim_t k,
    const void* alpha,
    const void* a,
    const void* b,
    const void* beta,
    void* c, inc_t rs_c0, inc_t cs_c0,
    const auxinfo_t* data, const cntx_t* cntx
);

% if "M" == vecdir:
extern void ${blis_ukr_name}_cs(dim_t m, dim_t n, dim_t k,
% elif "N" == vecdir:
extern void ${blis_ukr_name}_rs(dim_t m, dim_t n, dim_t k,
% endif
    const void* alpha,
    const void* a,
    const void* b,
    const void* beta,
    void* c, inc_t rs_c0, inc_t cs_c0,
    const auxinfo_t* data, const cntx_t* cntx
);
extern void ${blis_ukr_name}_csrs(dim_t m, dim_t n, dim_t k,
    const void* alpha,
    const void* a,
    const void* b,
    const void* beta,
    void* c, inc_t rs_c0, inc_t cs_c0,
    const auxinfo_t* data, const cntx_t* cntx
);

void ${blis_ukr_name}(dim_t m, dim_t n, dim_t k,
    const void* alpha,
    const void* a,
    const void* b,
    const void* beta,
    void* c, inc_t rs_c0, inc_t cs_c0,
    const auxinfo_t* data, const cntx_t* cntx)
{
    dim_t mr = ${mr};
    dim_t nr = ${nr};
    uint64_t rs_c = rs_c0;
    uint64_t cs_c = cs_c0;

    GEMM_UKR_SETUP_CT_ANY(${dtchar}, mr, nr, false)

    
% if "M" == vecdir:
    if(rs_c == 1 && cs_c == mr)
% elif "N" == vecdir:
    if(rs_c == nr && cs_c == 1)
% endif
    {
        ${blis_ukr_name}_nostride(
            m,n,k,
            alpha,a,b,
            beta,c,
            rs_c,cs_c,
            data,cntx);
    }
% if "M" == vecdir:
    else if(rs_c == 1)
    {
        ${blis_ukr_name}_cs(
% elif "N" == vecdir:
    else if(cs_c == 1)
    {
        ${blis_ukr_name}_rs(
% endif
            m,n,k,
            alpha,a,b,
            beta,c,
            rs_c,cs_c,
            data,cntx);
    }
    else
    {
        ${blis_ukr_name}_csrs(
            m,n,k,
            alpha,a,b,
            beta,c,
            rs_c,cs_c,
            data,cntx);
    }
    GEMM_UKR_FLUSH_CT(${dtchar})
    
}
"""
}
