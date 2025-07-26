import os
import torch
import torchmetrics
import argparse
import cddlt

from cddlt.datasets.rekis_dataset import ReKIS
from cddlt.datasets.cordex_dataset import CORDEX
from cddlt.dataloaders.downscaling_transform import DownscalingTransform

from cddlt.models.bicubic import Bicubic
from cddlt.models.srcnn import SRCNN
from cddlt.models.espcn import ESPCN
from cddlt.models.fno import FNO
# from cddlt.models.swinir import SwinIR

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--epochs", default=3, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--threads", default=4, type=int)
parser.add_argument("--upscale_factor", default=10, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--logdir", default="logs", type=str)
parser.add_argument("--variables", default=["TM"], type=list)

DATA_PATH = "..."

def main(args: argparse.Namespace) -> None:
    cddlt.startup(args)

    rekis = ReKIS(
        data_path=f"{DATA_PATH}/rekis",
        variables=args.variables,
        train_len=("2000-01-01", "2000-01-10"),
        dev_len=("2000-01-10", "2000-01-20"),
        test_len=("2000-01-20", "2000-02-01"),
        resampling="cubic_spline"
    )

    rekis_train = DownscalingTransform(dataset=rekis.train).dataloader(args.batch_size, shuffle=True)
    rekis_dev = DownscalingTransform(dataset=rekis.dev).dataloader(args.batch_size)

    cordex = CORDEX(
        data_path=f"{DATA_PATH}/cordex",
        variables=args.variables,
        dev_len=("2000-01-20", "2000-02-01"),
        test_len=("2000-02-01", "2000-03-01")
    )

    cordex_dev = DownscalingTransform(dataset=cordex.dev).dataloader(args.batch_size)
    cordex_test = DownscalingTransform(dataset=cordex.test).dataloader(args.batch_size)

    ### Bicubic

    bicubic = Bicubic(
        upscale_factor=args.upscale_factor
    )

    bicubic.configure(
        loss = torchmetrics.MeanSquaredError(squared=False),
        args = args,
        device = "cpu"
    )

    print(f"--- Bicubic ---")

    bicubic.evaluate(rekis_dev, print_loss=True, epochs=args.epochs)

    # ### SRCNN

    srcnn = SRCNN(
        n_channels=1,
        upscale_factor=args.upscale_factor,
    )

    srcnn.configure(
        optimizer = torch.optim.Adam(params=srcnn.parameters(), lr=args.lr),
        loss = torchmetrics.MeanSquaredError(squared=False),
        args = args,
        device = "cpu"
    )

    print(f"--- SRCNN ---")

    srcnn.fit(rekis_train, rekis_dev, args.epochs)

    srcnn.load_weights(os.path.join(args.logdir, srcnn.model_name))
    srcnn.evaluate(rekis_dev, print_loss=True)

    ### ESPCN

    espcn = ESPCN(
        n_channels = 1,
        upscale_factor = args.upscale_factor
    )

    espcn.configure(
        optimizer = torch.optim.Adam(params=espcn.parameters(), lr=args.lr),
        scheduler = None,
        loss = torchmetrics.MeanSquaredError(squared=False),
        args = args
    )

    print(f"--- ESPCN ---")

    espcn.fit(rekis_train, rekis_dev, args.epochs)

    espcn.load_weights(os.path.join(args.logdir, espcn.model_name))
    espcn.evaluate(rekis_dev, print_loss=True)

    ### FNO

    fno = FNO(
        n_channels=1,
        upscale_factor=args.upscale_factor
    )

    fno.configure(
        optimizer = torch.optim.Adam(params=fno.parameters(), lr=args.lr),
        scheduler = None,
        loss = torchmetrics.MeanSquaredError(squared=False),
        args = args,
        device="cpu"
    )

    print(f"--- FNO ---")

    fno.fit(rekis_train, rekis_dev, args.epochs)

    fno.load_weights(os.path.join(args.logdir, fno.model_name))
    fno.evaluate(rekis_dev, print_loss=True)

    ### cordex eval

    dev_predict = fno.predict(cordex_dev)
    test_predict = fno.predict(cordex_test)

    print(f"cordex dev predict shape: {dev_predict[0].shape}")
    print(f"cordex test predict shape: {test_predict[0].shape}")

    ### SwinIR

    # swin = SwinIR(
    #     img_size=40,
    #     in_chans=1,
    #     upscale=10 ## fix, implementaiton supports only 3 / 2^n upsacling factors
    # )

    # swin.configure(
    #     optimizer = torch.optim.Adam(params=fno.parameters(), lr=args.lr),
    #     scheduler = None,
    #     loss = torchmetrics.MeanSquaredError(squared=False),
    #     logdir = args.logdir,
    #     device="cpu"
    # )

    # print(f"--- SWIN ---")

    # swin.fit(rekis_train, rekis_dev, args.epochs)

    # swin.load_weights(args.logdir)
    # swin.evaluate(rekis_dev, log_loss=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)