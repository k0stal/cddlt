import numpy as np
import pandas as pd
import xarray as xr
import os

def generate_random_climate_data(
    output_path: str,
    start_year: int = 2000,
    end_year: int = 2002,
    n_northing: int = 400,
    n_easting: int = 400,
):
    time = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    easting = np.linspace(4.336e6, 4.752e6, n_easting)
    northing = np.linspace(5.954e6, 5.554e6, n_northing)

    coords = {
        "time": time,
        "northing": northing,
        "easting": easting
    }

    variables = {
        "PR": {"units": "mm", "long_name": "Precipitation", "range": (0, 50)},
        "TM": {"units": "°C", "long_name": "Mean Temperature", "range": (-10, 35)},
        "TN": {"units": "°C", "long_name": "Min Temperature", "range": (-20, 25)},
        "TX": {"units": "°C", "long_name": "Max Temperature", "range": (-5, 45)},
    }

    data_vars = {}
    shape = (len(time), n_northing, n_easting)

    print(f"Generating sample datset...")

    for var, meta in variables.items():
        low, high = meta["range"]
        data = np.random.uniform(low, high, size=shape).astype("float32")

        da = xr.DataArray(
            data,
            coords=coords,
            dims=["time", "northing", "easting"],
            name=var,
            attrs={
                "units": meta["units"],
                "long_name": meta["long_name"]
            }
        )
        data_vars[var] = da

    crs = xr.DataArray(
        data=-2147483647,
        attrs={"grid_mapping_name": "transverse_mercator"},
    )

    ds = xr.Dataset(data_vars)
    ds["crs"] = crs

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_netcdf(output_path, format="NETCDF4")
    print(f"Saved dummy dataset to: {output_path}")


def delete_dataset(path: str):
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted: {path}")
    else:
        print(f"File not found: {path}")

def main():
    output_path = ...
    generate_random_climate_data(output_path)

if __name__ == "__main__":
    main()
