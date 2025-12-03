isa_flags = {
    "rvv" : "-march=rv64gcv",
    "rvv071" : "-mepi",
    "neon" : "-march=armv8-a+simd",
    "sve" : "-march=armv8-a+sve",
    "sme" : "-march=armv8-a+sme+sme-f64f64",
    "avx128" : "-mavx -mfma",
    "avx256" : "-mavx2 -mfma",
    "avx512" : "-mavx512f -mavx512dq -mavx512",
}
