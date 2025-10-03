def get_ukr_components(ukr : str) -> list[str]:
    if "gemm" == ukr:
        return ["A","B","AB","C"]
    if "mm" == ukr:
        return ["A","B","C"]

    raise ValueError(f"Invalid microkernel {ukr}")
