#!/usr/bin/env python
# coding: utf-8

import os
import logging
import pickle
import sqlite3

import numpy as np
import pandas as pd
import h5py

from screpairutils import load_posarray_and_mappability

from msr.msr import MSR
from msr.segmentgraph import segment_map_to_graph, prune_segment_graph, flatten_segment_graph, validate_single_node_per_position, NoUnprunedNodes

from screpairutils.literals import DAMID_LIBTYPES, CHIC_LIBTYPES
from screpairutils.load_data import load_binned_ds, damid_fn_from_row, chic_fn_from_row
from screpairutils.scalingfactor import ds_convolve, calc_scale_factor, get_scaling_factor_kernel
from screpairutils.chromdicttools import map_dict, mask_ds, sum_ds


log = logging.getLogger(__name__)
logsh = logging.StreamHandler()
logsh.setLevel(logging.DEBUG)
logfmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
logsh.setFormatter(logfmt)
log.addHandler(logsh)


MAX_STDDEV = 10e6

SFS_ALPHA = 0.6

CUTOFFS = {
    "DamID": 500,
    "ChIC": 500,
}

# from: http://localhost:8080/notebooks/KR20200304.generating_ensemble_backgrounds.ipynb
GROUPING_COLUMNS = {
    "DamID": ["clone_id", "cellcount", ],
    "ChIC": ["antibody", "cellcount", ],
}

SAMPLETYPE2LIBTYPES = {
    "DamID": DAMID_LIBTYPES,
    "ChIC": CHIC_LIBTYPES,
}

SAMPLETYPE2COLPREFIX = {
    "DamID": "damid",
    "ChIC": "chic"
}

# per-sample output serialization key
KEY_COLS = ["limsid", "indexnr", "barcodenr"]
OUTFNFMT = "./output/KR20200729.MSR_output/KR20200729.MSR_results.binsize_{binsize:d}.worker_{i:03d}.db"


def to_task_count(tasks, workers):
    assert tasks > 0
    assert workers > 0

    workers_adj = min(tasks, workers)
    base_taskcount = tasks // workers_adj
    task_remainder = tasks % workers_adj

    assert task_remainder < workers_adj
    assert tasks == (base_taskcount * workers_adj) + task_remainder

    return (workers_adj, base_taskcount, task_remainder)


def task_index(worker_index, base_taskcount, task_remainder):
    return min(worker_index, task_remainder) + base_taskcount * worker_index


def stepvector_to_tsv(sv, binned_chromsizes):
    return pd.DataFrame(
        [
            (stepstart, stepend) + next(iter(stepvalues))
            for (stepstart, stepend, stepvalues) in sv[0:binned_chromsizes['1']]
            if any(stepvalues)
        ],
        columns=["startbin", "endbin", "level", "isegment"],
    ).to_csv(sep="\t", index=False, header=True).encode("utf8")


def pickledict(d):
    return [
        (pickle.dumps(k), pickle.dumps(v))
        for (k, v) in d.items()
    ]


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
    ap.add_argument('-o', '--outfnfmt', required=False, default=OUTFNFMT)
    ap.add_argument('annofn')
    ap.add_argument('manifest')
    ap.add_argument('bgfn')
    ap.add_argument('binsize', type=int)
    ap.add_argument('--vmr', type=float, required=False, default=1.)
    ap.add_argument('--verbose', '-v', action='count', required=False, default=0)
    ap.add_argument('--quiet', '-q', action="store_true", required=False, default=False)

    args = ap.parse_args()

    if args.quiet:
        log.setLevel(logging.CRITICAL)
    else:
        log.setLevel([logging.WARNING, logging.INFO, logging.DEBUG][min(2, args.verbose)])

    # get SLURM array job info
    n_workers = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    worker_index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    if worker_index >= n_workers:
        raise ValueError('No work for me! :D')

    binsize = args.binsize
    if binsize < 0:
        raise ValueError('Invalid binsize: %d' % binsize)

    # setup output database
    outfn = args.outfnfmt.format(
        binsize=binsize,
        i=worker_index,
    )
    assert not os.path.exists(outfn)
    dbcon = sqlite3.connect(outfn)
    dbcon.executescript("""
    BEGIN TRANSACTION;

    CREATE TABLE "sampledata" (
        key INTEGER PRIMARY KEY,
        sampletype TEXT NOT NULL,
        limsid VARCHAR NOT NULL,
        indexnr INT NOT NULL,
        barcodenr INT NOT NULL
    );
    CREATE UNIQUE INDEX sampledata_key_index ON "sampledata" (
        "sampletype",
        "limsid",
        "indexnr",
        "barcodenr"
    );

    CREATE TABLE "sampledata_chrom" (
        key INTEGER PRIMARY KEY,
        sampledata_key INTEGER,
        chrom TEXT,
        segment_map_s BLOB,
        tbl_s BLOB
    );
    CREATE UNIQUE INDEX sampledata_chrom_key_index ON "sampledata_chrom" (
        "sampledata_key",
        "chrom"
    );

    COMMIT;
    """)

    (
        chroms, pos, chromsizes,
        binned_pos, binned_chromsizes,
        binned_mapab, w_mapab,
    ) = load_posarray_and_mappability.setup(binsize)

    # load annotations
    # need columns to identify fn of sample
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

    samplekey2annoirows = anno.groupby(["limsid", "indexnr", "barcodenr"]).groups

    workload_df = pd.read_csv(args.manifest, sep="\t", index_col=0, na_values="N/A")
    workload_df = workload_df.fillna({"bg_key_s": "N/A"}).sample(frac=1., random_state=42)
    n_tasks = len(workload_df)
    n_workers_adj, base_taskcount, task_remainder = to_task_count(n_tasks, n_workers)
    task_start = task_index(worker_index, base_taskcount, task_remainder)
    task_end = task_index(worker_index + 1, base_taskcount, task_remainder)
    task_df = workload_df.iloc[task_start:task_end]

    # init data for data loading / scaling factor calculation
    w_mask = map_dict(np.logical_not, w_mapab)

    W_sf = get_scaling_factor_kernel(binsize)

    # sentinel to use mappability as background (when not suitable background in DB)
    use_mapab = None
    # cache of pre-loaded and pre-convolved backgrounds
    bg_cache = {}
    binned_mapab_ = map_dict(mask_ds, binned_mapab, w_mask)
    # introduce copy numbers of U2OS (female/no Y, X is single-copy)
    # NOTE that several (subchromosomal, subclonal(?)) CNVs are not taken into
    # account here
    u2os_cnvs = {chrom: 2 for chrom in chroms}
    u2os_cnvs["X"] = 1
    binned_mapab_ = {
        chrom: (u2os_cnvs[chrom] * binned_mapab_[chrom])
        for chrom in chroms
    }
    binned_mapab_c = ds_convolve(binned_mapab_, W_sf)
    binned_mapab_total = sum_ds(binned_mapab_)
    bg_cache[use_mapab] = (
        binned_mapab_,
        binned_mapab_c,
        binned_mapab_total,
    )

    # store results per irow:
    # per chrom:
    # - serialized MSR
    # - serialized stepvector of pruned graph
    # - SFC score after pruning
    # - actual FC after pruning

    # actual per-sample loop:
    for irow, row in task_df.iterrows():
        log.debug("Running irow: %d" % irow)

        results = {}

        # load bg
        bg_key_s = row['bg_key_s']
        if bg_key_s == "N/A":
            bg_key_s = None

        if bg_key_s not in bg_cache:
            with h5py.File(args.bgfn, 'r') as f:
                bg = {chrom: f[bg_key_s][chrom][:] for chrom in chroms}
                bg = map_dict(mask_ds, bg, w_mask)
                bg_c = ds_convolve(bg, W_sf)
                bg_total = sum_ds(bg)
                bg_cache[bg_key_s] = (bg, bg_c, bg_total)

        bg, bg_c, bg_total = bg_cache[bg_key_s]

        sampletype = row['sampletype']

        # need anno
        # need metadata in anno cols (barcodetype, etc)
        # to load fn
        # facepalm
        limsid, indexnr, barcodenr = (row['limsid'], row['indexnr'], row['barcodenr'])
        annoirows = samplekey2annoirows[(limsid, indexnr, barcodenr)]
        assert len(annoirows) == 1
        annoirow = next(iter(annoirows))
        annorow = anno.loc[annoirow]

        # load fg
        if sampletype == "DamID":
            assert annorow["library_type"] in DAMID_LIBTYPES
            fn = damid_fn_from_row(annorow, binsize=binsize)
            if not os.access(fn, os.R_OK):
                log.warning("Can't read file: %s" % fn)
                continue
        elif sampletype == "ChIC":
            assert annorow["library_type"] in CHIC_LIBTYPES
            try:
                fn = chic_fn_from_row(annorow, binsize=binsize)
            except ValueError as e:
                log.warning(e)
                continue

        fg = load_binned_ds(fn, binned_chromsizes=binned_chromsizes)
        fg = map_dict(mask_ds, fg, w_mask)
        fg_c = ds_convolve(fg, W_sf)

        # Calculate SF wrt ensemble background:
        Xij = np.array([
            np.concatenate([fg_c[chrom][w_mapab[chrom]] for chrom in chroms]),
            np.concatenate([bg_c[chrom][w_mapab[chrom]] for chrom in chroms]),
        ]).T
        rel_sf, sf_converged = calc_scale_factor(Xij, alpha=SFS_ALPHA)
        assert np.isclose(1., rel_sf.sum())
        sf = rel_sf[0] / (rel_sf[1] / bg_total)
        sf = max(250., sf)

        # run MSR
        m = MSR(
            fg,
            bg,
            scaling_factor=sf / bg_total,
            max_stddev_bins=int(np.ceil(MAX_STDDEV / binsize)),
            mapab=w_mapab,
            segmentation_pvalue=0.05,
            enrichment_pvalue=1e-5,
            vmr=args.vmr,
        )

        segment_maps = m.get_results()

        datacols = ["ilevel", "isegment"] + list(next(iter(next(iter(segment_maps.values())))).data.keys())  # lol

        for chrom in chroms:
            segment_map = segment_maps[chrom]

            # get segment graph
            segment_graph = segment_map_to_graph(segment_map)
            # prune it (NB: it prunes in-place)
            prune_segment_graph(segment_graph, segment_map.segment_counts(), T=1.05)

            # get segments on flat space:
            try:
                tree, unpruned_nodes = flatten_segment_graph(segment_graph)
                assert validate_single_node_per_position(tree)
                outtbl = pd.DataFrame([
                    [ilevel, isegment] + [segment_map[ilevel].data[col][isegment] for col in datacols[2:]]
                    for (ilevel, isegment) in unpruned_nodes
                ], columns=datacols)
            except NoUnprunedNodes:
                # empty table
                outtbl = pd.DataFrame([], columns=datacols)

            results[chrom] = (
                segment_map.serialize(),
                outtbl.to_csv(sep="\t", index=False, header=True).encode("utf8"),
            )

        with dbcon:
            dbcursor = dbcon.cursor()
            dbcursor.execute("""
            INSERT INTO "sampledata" (
                sampletype,
                limsid,
                indexnr,
                barcodenr
            ) VALUES (?, ?, ?, ?);
            """, (sampletype, limsid, indexnr, barcodenr)
            )
            sampledata_pk = dbcursor.lastrowid

            for chrom in chroms:
                dbcursor.execute("""
                INSERT INTO "sampledata_chrom" (
                    sampledata_key,
                    chrom,
                    segment_map_s,
                    tbl_s
                ) VALUES (?, ?, ?, ?);
                """, (sampledata_pk, chrom, results[chrom][0], results[chrom][1])
                )


if __name__ == "__main__":
    main()
