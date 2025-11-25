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

    % for kind,type,name in kernels:
        BLIS_${kind}_UKR, BLIS_${type}, ${name},
    % endfor

        BLIS_VA_END
    );
""",
    "cntx_set_kernel_prefs" : """
    bli_cntx_set_ukr_prefs(
        cntx,

    % for kind,type,_ in kernels:
        BLIS_${kind}_UKR_ROW_PREF, BLIS_${type}, FALSE,
    % endfor

        BLIS_VA_END
    );
""",
    "cntx_set_blocksizes" : """

    const uint32_t mr_d = ${simd_size}/sizeof(double);
    const uint32_t mr_s = ${simd_size}/sizeof(float);
    const uint32_t nr_d = ${nr_d};
    const uint32_t nr_s = ${nr_s};

    const uint32_t kc_d = ${kunroll_d}*32;
    const uint32_t kc_s = ${kunroll_s}*32;
    const uint32_t nc_d = 200*nr_d;
    const uint32_t nc_s = 200*nr_s;

    // Initialize level-3 blocksize objects with architecture-specific values.
    //                                              s        d        c        z
    bli_blksz_init_easy( &blkszs[ BLIS_MR ],     mr_s,    mr_d,      -1,      -1 );
    bli_blksz_init_easy( &blkszs[ BLIS_NR ],     nr_s,    nr_d,      -1,      -1 );
    bli_blksz_init_easy( &blkszs[ BLIS_MC ],  20*mr_s, 20*mr_d,      -1,      -1 );
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

patches = {
    "config_registry" :
    """
diff --git a/config_registry b/config_registry
index 81543934..0f6bec7f 100644
--- a/config_registry
+++ b/config_registry
@@ -65,5 +65,8 @@ rv64iv:      rv64iv/rviv
 sifive_rvv: sifive_rvv
 sifive_x280: sifive_x280/sifive_rvv
 
+# generated
+${configname}: ${configname}
+
 # Generic architectures.
 generic:     generic
""",

    "frame/base/bli_arch.c" :
    """
diff --git a/frame/base/bli_arch.c b/frame/base/bli_arch.c
index 776bb698..5224f99a 100644
--- a/frame/base/bli_arch.c
+++ b/frame/base/bli_arch.c
@@ -322,6 +322,11 @@ arch_t bli_arch_query_id_impl( void )
 		id = BLIS_ARCH_SIFIVE_X280;
 		#endif
 
+		// Generated microarchitecture.
+		#ifdef BLIS_FAMILY_${configname.upper()}
+		id = BLIS_ARCH_${configname.upper()};
+		#endif
+
 		// Generic microarchitecture.
 		#ifdef BLIS_FAMILY_GENERIC
 		id = BLIS_ARCH_GENERIC;
@@ -390,6 +395,8 @@ static const char* config_name[ BLIS_NUM_ARCHS ] =
     "sifive_rvv",
     "sifive_x280",
 
+    "${configname}",
+
     "generic"
 };
 
""",

    "frame/include/bli_arch_config.h" :
    """
diff --git a/frame/include/bli_arch_config.h b/frame/include/bli_arch_config.h
index 49a89430..240460d6 100644
--- a/frame/include/bli_arch_config.h
+++ b/frame/include/bli_arch_config.h
@@ -187,6 +187,12 @@ INSERT_GENTCONF
 #include "bli_family_sifive_x280.h"
 #endif
 
+// -- ukrgen --
+
+#ifdef BLIS_FAMILY_${configname.upper()}
+#include "bli_family_${configname}.h"
+#endif
+
 // -- Generic --
 
 #ifdef BLIS_FAMILY_GENERIC
@@ -287,6 +293,10 @@ INSERT_GENTCONF
 #include "bli_kernels_sifive_x280.h"
 #endif
 
+#ifdef BLIS_KERNELS_${configname.upper()}
+#include "bli_kernels_${configname}.h"
+#endif
+
 
 #endif
 
""",
    "frame/include/bli_gentconf_macro_defs.h" :
    """
diff --git a/frame/include/bli_gentconf_macro_defs.h b/frame/include/bli_gentconf_macro_defs.h
index f6f3af20..54fa99ac 100644
--- a/frame/include/bli_gentconf_macro_defs.h
+++ b/frame/include/bli_gentconf_macro_defs.h
@@ -233,6 +233,14 @@
 #define INSERT_GENTCONF_SIFIVE_X280
 #endif
 
+// -- Generated architectures --------------------------------------------------
+
+#ifdef BLIS_CONFIG_${configname.upper()}
+#define INSERT_GENTCONF_${configname.upper()} GENTCONF( ${configname.upper()}, ${configname} )
+#else
+#define INSERT_GENTCONF_${configname.upper()}
+#endif
+
 // -- Generic architectures ----------------------------------------------------
 
 #ifdef BLIS_CONFIG_GENERIC
@@ -288,7 +296,9 @@ INSERT_GENTCONF_RV64IV ${"\\\\"}
 INSERT_GENTCONF_SIFIVE_RVV ${"\\\\"}
 INSERT_GENTCONF_SIFIVE_X280 ${"\\\\"}
 ${"\\\\"}
-INSERT_GENTCONF_GENERIC
+INSERT_GENTCONF_${configname.upper()} ${"\\\\"}
+${"\\\\"}
+INSERT_GENTCONF_GENERIC 
 
 
 #endif
""",
    "frame/include/bli_type_defs.h" :
    """
diff --git a/frame/include/bli_type_defs.h b/frame/include/bli_type_defs.h
index 758f9eb3..2305c6a0 100644
--- a/frame/include/bli_type_defs.h
+++ b/frame/include/bli_type_defs.h
@@ -1010,6 +1010,9 @@ typedef enum arch_e
 	BLIS_ARCH_SIFIVE_RVV,
 	BLIS_ARCH_SIFIVE_X280,
 
+	// Generated architecture/configuration
+	BLIS_ARCH_${configname.upper()},
+
 	// Generic architecture/configuration
 	BLIS_ARCH_GENERIC,
 
"""

}


kernels = {
    "gemm" : """
#include "blis.h"

${ukr_extra_def}

extern void ${blis_ukr_name}_cs(dim_t m, dim_t n, dim_t k,
    const void* alpha,
    const void* a,
    const void* b,
    const void* beta,
    void* c, inc_t rs_c0, inc_t cs_c0,
    const auxinfo_t* data, const cntx_t* cntx
);
extern void ${blis_ukr_name}_rs(dim_t m, dim_t n, dim_t k,
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

    dim_t kiter = k / ${kunroll};

    if((kiter > 2) && rs_c == 1)
    {
        ${blis_ukr_name}_cs(
            m,n,kiter-1,
            alpha,a,b,
            beta,c,
            rs_c0,cs_c0,
            data,cntx);
    }
    else if((kiter > 2) && cs_c == 1)
    {
        ${blis_ukr_name}_rs(
            m,n,kiter-1,
            alpha,a,b,
            beta,c,
            rs_c0,cs_c0,
            data,cntx);
    }
    else if(kiter > 2)
    {
        ${blis_ukr_name}_csrs(
            m,n,kiter-1,
            alpha,a,b,
            beta,c,
            rs_c0,cs_c0,
            data,cntx);
    }
    GEMM_UKR_FLUSH_CT(${dtchar})
    
}
"""
}
