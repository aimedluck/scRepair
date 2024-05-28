import os
import sys
import sqlite3
import pickle
import base64

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from screpairutils.literals import DAMID_LIBTYPES, CHIC_LIBTYPES


SAMPLETYPES = ["DamID", "ChIC"]

CUTOFFS = {
    "DamID": 500,
    "ChIC": 500,
}

GROUPING_COLUMNS = {
    "DamID": ["clone_id", "cellcount", ],
    "ChIC": ["antibody_target", "cellcount", "antibody"],
}

SAMPLETYPE2LIBTYPES = {
    "DamID": DAMID_LIBTYPES,
    "ChIC": CHIC_LIBTYPES,
}

SAMPLETYPE2COLPREFIX = {
    "DamID": "damid",
    "ChIC": "chic"
}


def np2pytype(x):
    if isinstance(x, np.generic):
        return x.item()
    elif isinstance(x, np.ndarray):
        return x.tolist()
    else:
        return x


def main():
    """
    Why waste time write lot functions when few functions do trick?
    - Kevin
    """

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('annofn')
    ap.add_argument('databasefn')
    ap.add_argument('bgfn')
    ap.add_argument('binsize', type=int)

    args = ap.parse_args()

    binsize = args.binsize
    if binsize < 0:
        raise ValueError('Invalid binsize: %d' % binsize)

    dbfn = args.databasefn
    assert os.path.exists(dbfn)
    dbcon = sqlite3.connect(dbfn)
    # pre-cache all sample keys
    db_samples = set(dbcon.execute("""
    SELECT
        s.sampletype,
        s.limsid,
        s.indexnr,
        s.barcodenr
    FROM sampledata AS s;
    """).fetchall())

    # ### Load annotations
    anno = pd.read_csv(
        args.annofn,
        sep="\t",
        na_values="N/A",
        encoding="utf8",
        comment="#",
        low_memory=False,
    )

    # *Fix cellcount not being numeric anymore due to "bulk"*
    anno.loc[anno["cellcount"] == "bulk", "cellcount"] = 1_000_000
    anno["cellcount"] = anno["cellcount"].astype(int)

    assert not anno[["limsid", "indexnr", "barcodenr"]].isna().any().any()
    assert (anno.groupby(["limsid", "indexnr", "barcodenr"]).size().max() == 1)

    # load available backgrounds
    with h5py.File(args.bgfn, 'r') as f:
        bg_keys = {
            pickle.loads(base64.b32decode(k)): k
            for k in f.keys()
        }

    # for every sampletype
    # for every row that passes cutoffs
    # list workitem
    # - sampletype, limsid, indexnr, barcodenr
    # - matching background (encoded/pickled?)
    # - binsize?
    outrows = []
    for sampletype in SAMPLETYPES:
        workload_df = anno[
            anno["library_type"].isin(SAMPLETYPE2LIBTYPES[sampletype])
            & (anno["%s_total" % SAMPLETYPE2COLPREFIX[sampletype]] >= CUTOFFS[sampletype])
        ]

        cols = GROUPING_COLUMNS[sampletype]

        for irow, row in tqdm(workload_df.iterrows(), total=len(workload_df)):
            sample_key = (
                (sampletype, ) + tuple(map(np2pytype, (
                    row['limsid'], row['indexnr'], row['barcodenr'],
                )))
            )
            # check if sample already present in MSR output database
            if sample_key in db_samples:
                continue

            # find matching background in bg hierarchy
            try:
                bg_key = next(
                    x for (i, x) in (
                        (i, (sampletype, tuple(map(np2pytype, row[cols[:i]]))))
                        for i in range(len(cols), 0, -1)
                    )
                    if x in bg_keys
                )
            except StopIteration:
                bg_key = None

            outrows.append((
                row['limsid'],
                row['indexnr'],
                row['barcodenr'],
                sampletype,
                (bg_keys[bg_key] if bg_key is not None else 'N/A'),
            ))

    outtbl = pd.DataFrame(outrows, columns=['limsid', 'indexnr', 'barcodenr', 'sampletype', 'bg_key_s'])
    outtbl.to_csv(sys.stdout, sep="\t", index=True, index_label="irow", header=True)

    return


if __name__ == "__main__":
    main()
