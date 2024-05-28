import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import single, fcluster

from .literals import conf


def load_asisi_sites(chroms):
    asisi_sites = pd.read_csv(conf["asisi_site_fn"], sep="\t", header=None, names=["chrom", "start", "end"])
    # NOTE: asisi sites are also listed on contigs (GL.. etc); remove those entries
    asisi_sites = asisi_sites[asisi_sites["chrom"].isin(chroms)].reset_index(drop=True)

    # src:
    # 10.1016/j.molcel.2018.08.020 table S1
    s = """
chromosome start end  chromosome start end chr1 9649446 9649452  chr10 3110978 3110984 chr1 40974644 40974650  chr10 94051015 94051021 chr1 89458597 89458603  chr11 24518476 24518482 chr1 110036700 110036706  chr11 75525761 75525767 chr1 204380453 204380459  chr11 85375655 85375661 chr1 224032648 224032654  chr12 13154718 13154724 chr2 43358339 43358345  chr12 22093989 22093995 chr2 55509101 55509107  chr12 121975058 121975064 chr2 68384749 68384755  chr12 130091881 130091887 chr2 74734762 74734768  chr13 105238552 105238558 chr2 85822594 85822600  chr13 114894659 114894665 chr2 120124566 120124572  chr14 54955826 54955832 chr2 208030728 208030734  chr17 5390221 5390227 chr3 52232163 52232169  chr17 38137473 38137479 chr3 98618165 98618171  chr17 57184297 57184303 chr3 99536965 99536971  chr17 61850856 61850862 chr4 83934287 83934293  chr17 80250841 80250847 chr4 178363576 178363582  chr18 7566713 7566719 chr5 68462851 68462857  chr18 19320805 19320811 chr5 79784140 79784146  chr19 2456094 2456100 chr5 142785050 142785056  chr19 30019488 30019494 chr6 27145367 27145373  chr19 41903743 41903749 chr6 31105428 31105434  chr19 42497856 42497862 chr6 37321812 37321818  chr19 45932080 45932086 chr6 49917583 49917589  chr19 46768784 46768790 chr6 67704021 67704027  chr20 1207616 1207622 chr6 90348187 90348193  chr20 20032925 20032931 chr6 135819348 135819354  chr20 30946313 30946319 chr6 144607569 144607575  chr20 32032087 32032093 chr6 149888106 149888112  chr20 37360269 37360275 chr7 75807507 75807513  chr20 42087118 42087124 chr7 92861491 92861497  chr21 33245519 33245525 chr7 99679508 99679514  chr21 46221790 46221796 chr8 66546348 66546354  chr22 20850308 20850314 chr8 116680632 116680638  chr22 38864102 38864108 chr8 124781210 124781216  chrX 1510672 1510678 chr9 29212800 29212806  chrX 45366394 45366400 chr9 36258514 36258520  chrX 53111427 53111433 chr9 127532106 127532112  chrX 72783103 72783109 chr9 130693171 130693177         chr9 130889408 130889414
"""
    s_words = s.strip().split()[6:]  # skip the two headers (2x3 words)
    top_sites = pd.DataFrame({
        "chrom": [x.split("chr", 1)[1] for x in s_words[0::3]],
        # AFAICT the `start` is off by 2
        "start": [int(x) - 2 for x in s_words[1::3]],
        "end": [int(x) for x in s_words[2::3]],
    })

    # list for every site in `asisi_sites` whether it's a top site:
    top_site_tuples = set(top_sites.apply(lambda row: (row["chrom"], row["start"], row["end"]), axis=1))
    asisi_sites["is_top_site"] = asisi_sites.apply(lambda row: (row['chrom'], row['start'], row['end']) in top_site_tuples, axis=1)

    return asisi_sites, top_sites


###

# Code to "cluster" nearby sites
# Note that an inherent danger is that well-dispersed sites, in combination
# with a large `maxdist` will "glue together" a lot of sites, perhaps all
# across a single chrom

# Note that `PyRanges` doesn't have an API for what I want to do... `split()` comes close but doesn't cut it
# I could've used my `StepVector` but decided against yet another dependency


# Recipe:
#
# - group sites per chrom
# - use `fcluser(single(pdist(...)))` approach to merge sites within a distance; obtain (arbitrary) cluster IDs
# - group sites on cluster ID
# - per cluster:
#     - get overall cluster start and end
#     - get number of sites within cluster
#     - get sorted set of site positions (start, center, end?)
#     - get geometric mean of centers


def cluster_sites(c, maxdist):
    """
    Cluster sites at positions `c` that lie at most `maxdist` positions away from eachother
    """

    assert c.ndim == 1
    maxdist = float(maxdist)

    if len(c) == 1:
        return np.array([0])

    # note that cluster IDs from `fcluster` are 1-based
    return fcluster(single(pdist(np.atleast_2d(c).T, metric="euclidean")), t=maxdist, criterion="distance") - 1


def get_cluster_df(chromdf, maxdist):
    assert "chrom" in chromdf.columns
    assert len(set(chromdf["chrom"])) == 1

    assert "start" in chromdf.columns
    assert "end" in chromdf.columns

    site_centers = chromdf[["start", "end"]].sum(axis=1) // 2
    cluster_ids = cluster_sites(site_centers.values, maxdist)
    return pd.DataFrame.from_records(chromdf.groupby(cluster_ids).apply(
        lambda subdf: {
            "n_sites": len(subdf),
            "sites_start": tuple(sorted(subdf["start"])),
            "sites_irows": tuple(sorted(subdf.index)),
            "cluster_start": subdf["start"].min(),
            "cluster_end": subdf["end"].max(),
            "cluster_geometric_mean": np.exp(np.mean(np.log(subdf[["start", "end"]].mean(axis=1)))).round().astype(int),
        },
    )).sort_values("cluster_start").reset_index(drop=True)


def cluster_site_df(df, maxdist):
    assert all(col in df.columns for col in ("chrom", "start", "end"))

    return df.groupby("chrom").apply(
        lambda chromdf: get_cluster_df(chromdf, maxdist),
    ).reset_index().drop(columns=["level_1"])
