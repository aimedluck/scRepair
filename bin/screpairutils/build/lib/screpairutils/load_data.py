import os

import numpy as np
import h5py

from screpairutils.literals import (
    BARCODENAMEFMTS, DAMID_OLDPOPFMT_LIMSIDS,
    DAMID_BINNED_FNFMT, DAMID_BINNED_FNFMT_OLDPOP,
    CHIC_BINNED_FNFMTS,
)


def get_dataset_with_dim(f, name, shape, dtype=None):
    if name not in f:
        return np.zeros(shape, dtype=(np.int if dtype is None else dtype))
    else:
        d = f[name][:]
        if d.shape == shape:
            if dtype is None:
                return d
            else:
                return d.astype(dtype)

        if len(shape) > d.ndim:
            raise ValueError("Could be implemented with `np.pad` but I'm lazy")

        if d.ndim > len(shape):
            reduceaxes = d.ndim - len(shape)
            d = np.add.reduce(d, -reduceaxes)

        diff = shape[0] - len(d)
        if diff > 0:
            d = np.pad(d, (0, diff), mode='constant', constant_values=d.dtype.type())
        elif diff < 0:
            d = d[:shape[0]]

        return d


def barcode_from_row(row):
    return BARCODENAMEFMTS[row["genomic_barcodetype"]].format(barcodenr=row["barcodenr"])


def damid_fn_from_row(row, binsize):
    barcode = barcode_from_row(row)

    if (
        (row["limsid"] in DAMID_OLDPOPFMT_LIMSIDS)
        and (row["cellcount"] > 16)
    ):
        fn = DAMID_BINNED_FNFMT_OLDPOP.format(**row.to_dict(), barcode=barcode, binsize=binsize)
    else:
        fn = DAMID_BINNED_FNFMT.format(**row.to_dict(), barcode=barcode, binsize=binsize)

    return fn


def chic_fn_from_row(row, binsize):
    barcode = barcode_from_row(row)

    # deal with multiple output file formats :/
    fns = map(
        lambda fnfmt: fnfmt.format(**row.to_dict(), barcode=barcode, binsize=binsize),
        CHIC_BINNED_FNFMTS,
    )
    try:
        fn = next(fn for fn in fns if os.access(fn, os.R_OK))
    except StopIteration:
        raise ValueError("No filename for %s" % row)

    return fn


def load_binned_ds(fn, binned_chromsizes):
    chroms = binned_chromsizes.keys()

    with h5py.File(fn, 'r') as f:
        ds = {
            chrom: get_dataset_with_dim(f, chrom, (binned_chromsizes[chrom], ))
            for chrom in chroms
        }

    return ds
