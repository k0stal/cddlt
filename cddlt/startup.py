import os
import torch
import random
import argparse
import numpy as np

def startup(
    args: argparse.Namespace
) -> None:

    ### allow fast matrix multiplication
    torch.backends.cuda.matmul.allow_tf32 = True

    ### reproducibility
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    seed = args.seed
    threads = args.threads

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if threads is not None and threads > 0:
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != threads:
            torch.set_num_interop_threads(threads)