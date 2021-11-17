import h5py
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def LinearInterpolate(source, target):
    def __init__(self, source, target):
        # determine scaling from sizes
        scales = [t / s for s, t in zip(source.shape, target.shape)]
        # initialize a buffer holding the resampling grid
        g = F.affine_grid(theta=torch.tensor([[scales[0], 0, 0], [0, scales[1], 0],]),)
        self.register_buffer("grid", g)

    def forward(self, x):
        F.grid_sample(x, self.grid, align_corners=False)


class Gaussian1D(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        ks = int(sigma * 5)  # 5 sigma kernel size
        if ks % 2 == 0:  # ensure odd kernel size
            ks += 1
        ts = torch.linspace(-ks // 2, ks // 2 + 1, ks)
        gauss = torch.exp((-((ts / sigma) ** 2) / 2))
        gauss /= gauss.sum()
        self.padding = (ks - 1) // 2
        self.register_buffer("kernel", gauss)

    def forward(self, x):
        return F.conv1d(x, self.kernel, padding=self.padding)


class Gaussian2D(nn.Module):
    def __init__(self, sigma):
        """
        Args:
            sigma: Union[float, Tuple[float]]:
                Sigma of Gaussian in each dimension (YX ordering).
        """
        super().__init__()
        if not isinstance(sigma, (list, tuple)):
            sigma = (sigma, sigma)
        assert len(sigma) == 2

        self.convy = Gaussian1D(sigma[0])
        self.convx = Gaussian1D(sigma[1])

    def forward(self, x):
        # Perform convolution as series of 1D convs
        bx = self.convx(x)
        by = self.convy(bx.T).T
        return by


class GaussianDownscale(nn.Module):
    def __init__(self, source, target, scales):
        super().__init__()
        # Initialize Gaussian convolution submodule
        # determine scale and width of Gaussian
        self.gaussian = Gaussian2D(sigma=[2 / s for s in scales])
        # initialize a buffer holding the resampling grid
        g = F.affine_grid(
            theta=torch.tensor([[[scales[0], 0, 0], [0, scales[1], 0],]]),
            size=(1, *target.shape),
        )
        self.register_buffer("grid", g)

    def forward(self, x):
        # convolve by Gaussian
        # interpolate by grid
        return F.grid_sample(x, self.grid)


def regrid(source, target, device="cuda"):
    """
    Given two datasets, regrid source to match shape of target and overwrite it.

    We use the following strategies:
    
    (Downscaling) When the target shape is smaller than the source shape:
        - If the source shape is an integer multiple of the target shape, we perform average pooling.
        - Otherwise, we perform Gaussian blurring followed by linear interpolation.
    (Upscaling) When the target shape is larger than the source shape:
        - We perform linear interpolation.

    Args:
        source: array_like:
            Source 2D dataset of size ...CHW. This dataset will be regridded.
        target: array_like:
            Mutable 2D dataset of size ...CUV. This dataset will be written to
            with the regridded (interpolated or pooled) source data.
    """
    assert source.shape[:-2] == target.shape[:-2]
    assert source.ndim == 3, "N dimension not yet supported"

    scales = [t / s for s, t in zip(source.shape[-2:], target.shape[-2:])]

    # Determine what function to apply to each channel
    # are we upscaling or downscaling?
    down = all(sc < 1 for sc in scales)
    up = all(sc >= 1 for sc in scales)
    if down != (not up):
        raise RuntimeError(
            f"Only pure downscaling or pure upscaling are supported. Found scales: {scales}"
        )
    if up:
        # func = LinearInterpolate(source, target, scales).to(device)
        func = nn.Upsample(size=target.shape[-2:]).to(device)
    else:  # downscaling
        # determine whether we can get away with simple average pooling
        remainders = [s % t for s, t in zip(source.shape, target.shape)]
        if all(r == 0 for r in remainders):  # we can pool!
            pooling = [s // t for s, t in zip(source.shape, target.shape)]
            func = nn.AvgPool2d(pooling).to(device)
        else:
            # can't pool, so we need to do combined
            func = GaussianDownscale(source, target, scales).to(device)

    # At this point, func should take in a torch.Tensor of shape 1HW and
    # transform it to the target shape.

    # Now stream 2D gray images and pass each one to `func`, then write it out
    with torch.no_grad():
        # TODO: support dimensions preceding channel
        for c in tqdm(range(source.shape[-3]), desc="Regridding channels"):
            im = torch.tensor(source[[c], :, :]).unsqueeze(0).to(device)
            txim = func(im).squeeze(0).cpu()
            target[c, :, :] = txim


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_dataset",
        nargs=2,
        help="HDF5 filename followed by dataset name for source data.",
    )
    parser.add_argument(
        "target_dataset",
        nargs=2,
        help="HDF5 filename followed by dataset name to write. Must not exist.",
    )
    parser.add_argument(
        "--output_shape",
        nargs=2,
        type=int,
        help="2D shape of output array (ignoring channel count), in YX order.",
    )
    args = parser.parse_args()

    source_file, source_dsname = args.source_dataset
    target_file, target_dsname = args.target_dataset

    with h5py.File(source_file, "r") as fsource:
        ds_source = fsource[source_dsname]
        with h5py.File(target_file, "a") as ftarget:
            if target_dsname in ftarget:
                raise RuntimeError(
                    f"Refusing to overwriting existing dataset {target_dsname} in {target_file}"
                )
            ds_target = ftarget.create_dataset(
                target_dsname, (*ds_source.shape[:-2], *args.output_shape), dtype="f",
            )
            regrid(ds_source, ds_target)
