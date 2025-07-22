import os
import argparse
import shutil
import tempfile
import cddlt

from cddlt.datasets.rekis_dataset import ReKIS
from cddlt.datasets.cordex_dataset import CORDEX
from cddlt.dataloaders.downscaling_transform import DownscalingTransform

from generate_test_data import generate_random_climate_data, delete_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--epochs", default=2, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--threads", default=4, type=int)
parser.add_argument("--upscale_factor", default=10, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--logdir", default="logs", type=str)
parser.add_argument("--variables", default=["TM"], type=list)

def main(args: argparse.Namespace) -> None:
    cddlt.startup(args, os.path.basename(__file__))

    tmp_root = tempfile.mkdtemp(prefix="cddlt_test_")
    data_root = os.path.join(tmp_root, "data")
    rekis_dir = os.path.join(data_root, "ReKIS")
    cordex_dir = os.path.join(data_root, "CORDEX")
    os.makedirs(rekis_dir, exist_ok=True)
    os.makedirs(cordex_dir, exist_ok=True)

    try:
        # --- ReKIS dataset test ---
        rekis_path = os.path.join(rekis_dir, "data.nc")
        generate_random_climate_data(
            output_path=rekis_path,
            start_date="2000-01-01",
            end_date="2000-04-01",
            n_northing=400,
            n_easting=400
        )

        # rekis dataset for evaluation
        rekis = ReKIS(
            data_path=data_root,
            variables=args.variables,
            train_len=("2000-01-01", "2000-02-01"),
            dev_len=("2000-02-01", "2000-03-01"),
            test_len=("2000-03-01", "2000-04-01"),
            resampling="cubic_spline"
        )

        train = DownscalingTransform(rekis.train).dataloader(args.batch_size, shuffle=True)
        dev = DownscalingTransform(rekis.dev).dataloader(args.batch_size)

        delete_dataset(rekis_path)

        # --- CORDEX dataset test ---
        cordex_path = os.path.join(cordex_dir, "data.nc")
        generate_random_climate_data(
            output_path=cordex_path,
            start_date="2000-03-01",
            end_date="2000-06-01",
            n_northing=40,
            n_easting=40
        )

        cordex = CORDEX(
            data_path=data_root,
            variables=args.variables,
            dev_len=("2000-03-01", "2000-04-01"),
            test_len=("2000-04-01", "2000-06-01"),
            resampling="cubic_spline"
        )

        dev = DownscalingTransform(cordex.dev).dataloader(args.batch_size)
        test = DownscalingTransform(cordex.test).dataloader(args.batch_size)

        delete_dataset(cordex_path)

    finally:
        shutil.rmtree(tmp_root)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)