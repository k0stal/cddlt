import os
import re
import torch
import random
import argparse
import datetime
import numpy as np

"""
Maybe create logdir per model, not per startup?.
"""

def startup(
    args: argparse.Namespace,
    script_name: str
) -> None:

    ### allow fast matrix multiplication
    torch.backends.cuda.matmul.allow_tf32 = True

    ### reproducibility
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        script_name,
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

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