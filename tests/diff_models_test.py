import os
import torch
import torchmetrics
import argparse
import cddlt

from cddlt.datasets.rekis_dataset import ReKIS
from cddlt.datasets.cordex_dataset import CORDEX
from cddlt.dataloaders.downscaling_transform import DownscalingTransform

from cddlt.models.diffusion import Diffusion, residual_collate_fn, residual_transform

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--threads", default=6, type=int)
parser.add_argument("--upscale_factor", default=10, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--logdir", default="logs", type=str)
parser.add_argument("--variables", default=["TM"], type=list)

DATA_PATH = "/Users/petr/Documents/projects/cddlt/data"

def main(args: argparse.Namespace) -> None:
    cddlt.startup(args)

    ### Rekis

    rekis = ReKIS(
        data_path=f"{DATA_PATH}/rekis",
        variables=args.variables,
        train_len=("2000-01-01", "2000-01-5"),
        dev_len=("2000-01-5", "2000-01-10"),
        test_len=("2000-01-10", "2000-01-15"),
        resampling="cubic_spline",
        standardize=True,
        residual=True
    )

    rekis_train = DownscalingTransform(
        dataset=rekis.train,
        collate_fn=residual_collate_fn,
        transform=residual_transform
    ).dataloader(args.batch_size, shuffle=True)
    rekis_dev = DownscalingTransform(
        dataset=rekis.dev,
        collate_fn=residual_collate_fn,
        transform=residual_transform
    ).dataloader(args.batch_size)

    ### Cordex

    cordex = CORDEX(
        data_path=f"{DATA_PATH}/cordex",
        variables=args.variables,
        dev_len=("2000-01-20", "2000-02-01"),
        test_len=("2000-02-01", "2000-03-01"),
        standardize=True,
        standard_params=rekis.standard_params,
        residual=True
    )

    cordex_dev = DownscalingTransform(
        dataset=cordex.dev,
        collate_fn=residual_collate_fn,
        transform=residual_transform
    ).dataloader(args.batch_size)

    cordex_test = DownscalingTransform(
        dataset=cordex.test,
        collate_fn=residual_collate_fn,
        transform=residual_transform
    ).dataloader(args.batch_size)

    ### Diffusion

    diff = Diffusion(
        n_channels = 1,
        params = rekis.standard_params["targets"]["TM"],
        num_steps=2
    )

    diff.configure(
        optimizer = torch.optim.Adam(params=diff.parameters(), lr=args.lr),
        loss = torchmetrics.MeanSquaredError(squared=False),
        args = args,
        metrics={
            "mse": torchmetrics.MeanSquaredError(squared=True),
            "mae": torchmetrics.MeanAbsoluteError(),
            "psnr": torchmetrics.image.PeakSignalNoiseRatio(),
        },
        device = "cpu"
    )

    print(f"--- DIFF ---")    

    diff.fit(rekis_train, rekis_dev, args.epochs)

    diff.load_weights(os.path.join(args.logdir, diff.model_name))
    diff.evaluate(rekis_dev, print_loss=True)

    dev_predict = diff.predict(cordex_dev)
    test_predict = diff.predict(cordex_test)

    print(f"cordex dev predict shape: {dev_predict[0].shape}")
    print(f"cordex test predict shape: {test_predict[0].shape}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)