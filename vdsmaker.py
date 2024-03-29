import h5py
import numpy
import shutil
from itertools import groupby


def split_big_data(nxs_in, n):

    # first interrogate the full data set to get the complete info
    subfile_size = {}
    omega = None
    underlying_dtype = None

    with h5py.File(nxs_in, "r") as f:
        omega = f["/entry/data/omega"][()]
        omega_end = f["/entry/sample/transformations/omega_end"][()]
        omega_increment_set = f["/entry/sample/transformations/omega_increment_set"][()]
        for k in f["/entry/data"]:
            if k.startswith("data_"):
                subfilename = f["/entry/data"][k].file.filename
                shape = f["/entry/data"][k].shape
                if underlying_dtype is None:
                    underlying_dtype = f["/entry/data"][k].dtype
                subfile_size[subfilename] = shape

    blocks = [subfile_size[s][0] for s in sorted(subfile_size)]
    nn = sum(([(i, j) for i in range(n)] for j, n in enumerate(blocks)), [])
    nframes = len(nn)

    subfilenames = list(sorted(subfile_size))

    assert nframes % n == 0

    bs = nframes // n

    # figure out which data sets consist of which blocks of data

    fmt = "%%0%dd" % len(str(n))

    for j in range(n):
        start = bs * j
        end = bs * (j + 1)
        chunk = nn[start:end]

        nxs_out = nxs_in.replace(".nxs", "_%s.nxs" % (fmt % (j + 1)))
        print("%s: %d -> %d" % (nxs_out, start, end))
        shutil.copyfile(nxs_in, nxs_out)

        with h5py.File(nxs_out, "r+") as f:
            f["/entry/data/omega"].resize([end - start])
            f["/entry/data/omega"][...] = omega[start:end]
            f["/entry/sample/transformations/omega_end"].resize([end - start])
            f["/entry/sample/transformations/omega_end"][...] = omega_end[start:end]
            f["/entry/sample/transformations/omega_increment_set"].resize([end - start])
            f["/entry/sample/transformations/omega_increment_set"][
                ...
            ] = omega_increment_set[start:end]

            for k in f["/entry/data"]:
                if k.startswith("data_"):
                    del f["/entry/data"][k]

            # make a VDS for this block of data - N.B. that this will cross
            # data block boundaries in the general case

            vds = h5py.VirtualLayout(
                shape=(end - start,) + shape[1:], dtype=underlying_dtype
            )

            # now work out
            last = 0
            for block, indices in groupby(chunk, lambda x: x[1]):
                group = list(indices)
                block_start = group[0][0]
                block_end = group[-1][0] + 1

                # this has to be the full data block
                subfilename = subfilenames[block]
                block_shape = subfile_size[subfilename]
                data = h5py.VirtualSource(subfilename, "data", shape=block_shape)
                data = data[block_start:block_end, :, :]

                # then assign it
                vds[last : last + (block_end - block_start), :, :] = data
                last += block_end - block_start

            # delete the data entry - if necessary
            if "/entry/data/data" in f:
                del f["/entry/data/data"]

            # replace with the VDS
            f.create_virtual_dataset("/entry/data/data", vds, fillvalue=-1)


if __name__ == "__main__":
    import sys

    nxs_in = sys.argv[1]
    n = int(sys.argv[2])

    split_big_data(nxs_in, n)
