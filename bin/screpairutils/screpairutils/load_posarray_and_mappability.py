#!/usr/bin/env python
# coding: utf-8
import re
import logging

import h5py
import numpy as np
import pandas as pd

from screpairutils.literals import conf

log = logging.getLogger(__name__)


def is_valid_chrom(s):
    return (s not in {"MT", }) and (s[:4] != "ERCC") and (s[:2] not in {"GL", "JH"})


def natural_sort_key(s):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]


# ### Load GATC positions and mappability data, and bin the mappability

# #### BONUS: mask "problematic" regions ("DAC Blacklisted Regions")
#
# src: https://www.nature.com/articles/s41598-019-45839-z -> https://github.com/Boyle-Lab/Blacklist/ -> https://www.encodeproject.org/annotations/ENCSR636HFF/

POSFN = conf['posfn']
MAPFN = conf['mapfn']
# TODO find alternative nomenclature... not happy with allowlist/denylist for this usecase tho..
BLACKLISTFN = conf['blacklistfn']


def load_pos_and_chromsizes():
    """
    NB: chromsizes are based on last pos for each chrom!
    """
    with h5py.File(POSFN, 'r') as f:
        chroms = sorted(filter(is_valid_chrom, f.keys()), key=natural_sort_key)
        pos = {chrom: f[chrom][:].cumsum() for chrom in chroms}
        chromsizes = {chrom: int(pos[chrom][-1]) + 1 for chrom in chroms}

    return chroms, pos, chromsizes


def bin_pos_and_chromsizes(chroms, pos, chromsizes, binsize):
    binned_chromsizes = {
        chrom: int(np.ceil(chromsizes[chrom] / binsize))
        for chrom in chroms
    }

    binned_pos = {chrom: (pos[chrom] // binsize) for chrom in chroms}

    return binned_pos, binned_chromsizes


def load_mappability(chroms):
    with h5py.File(MAPFN, 'r') as f:
        mapab = {chrom: (f[chrom][:] > 0) for chrom in chroms}

    return mapab


def load_blacklisttable(chroms):
    blacklisttbl = pd.read_csv(BLACKLISTFN, sep="\t", compression="gzip", header=None)

    blacklisttbl.columns = ["chrom", "start", "end", "type", "score", "strand"]

    assert blacklisttbl["chrom"].apply(lambda s: s.startswith("chr")).all()  # :(

    CHRNAME_MAP = {
        chrom: "chr%s" % chrom
        for chrom in chroms
        if chrom not in "MT"
    }

    CHRNAME_MAP['MT'] = "chrM"

    INV_CHRNAME_MAP = {
        v: k for (k, v) in CHRNAME_MAP.items()
    }

    blacklisttbl["chrom"] = blacklisttbl["chrom"].map(INV_CHRNAME_MAP)

    return blacklisttbl


def mask_blacklisted(chroms, pos, chromsizes, mapab, blacklisttable, padding=100):
    """
    NB: modifies `mapab` in-place
    """
    padding = int(padding)
    assert padding >= 0

    for chrom, chromsubdf in blacklisttable.groupby(["chrom"]):
        if chrom not in chroms:
            log.warn("Skipping %s" % chrom)
            continue

        segments = np.array([
            # add some padding since GATC may fall out of blacklisted region but majority of read sequence may fall into it
            np.searchsorted(
                pos[chrom],
                np.maximum(0, chromsubdf["start"].values - padding).astype(int),
            ),
            np.searchsorted(
                pos[chrom],
                np.minimum(chromsizes[chrom], chromsubdf["end"].values + padding).astype(int),
            ),
        ]).T

        for start, end in segments:
            mapab[chrom][start:end] = False

    return  # NB: modifies mapab in-place


def bin_mappability(chroms, mapab, binned_pos, binned_chromsizes):
    binned_mapab = {chrom: np.zeros(binned_chromsizes[chrom], dtype=int) for chrom in chroms}
    for chrom in chroms:
        np.add.at(binned_mapab[chrom], binned_pos[chrom], mapab[chrom].sum(axis=-1).astype(int))

    return binned_mapab


def get_cutoff_mapab(binsize):
    return 2 * (binsize / 1000)  # 2 per kb.


def binned_mappability_mask(chroms, binned_mappability, cutoff_mapab):
    return {chrom: binned_mappability[chrom] >= cutoff_mapab for chrom in chroms}


def setup(binsize):
    """
    Easy wrapper meant to give you the usuals:
        - chroms and chromsizes
        - GATC positions, mappability
            - applies blacklisting
        - binned versions of everything
    """

    chroms, pos, chromsizes = load_pos_and_chromsizes()
    binned_pos, binned_chromsizes = bin_pos_and_chromsizes(chroms, pos, chromsizes, binsize)
    mapab = load_mappability(chroms)
    blacklisttbl = load_blacklisttable(chroms)
    mask_blacklisted(chroms, pos, chromsizes, mapab, blacklisttbl)
    binned_mapab = bin_mappability(chroms, mapab, binned_pos, binned_chromsizes)
    cutoff_mapab = get_cutoff_mapab(binsize)
    w_mapab = binned_mappability_mask(chroms, binned_mapab, cutoff_mapab)

    return (
        chroms, pos, chromsizes,
        binned_pos, binned_chromsizes,
        binned_mapab, w_mapab
    )
