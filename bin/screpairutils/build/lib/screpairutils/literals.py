import os
from collections import ChainMap


# load configuration state
# and have some reasonable fallback defaults
DEFAULT_CONF = {
    "PREFIX": "./",
    "posfn": (
        "/data/zfs/references/"
        "human/hg19/posarray/"
        "Homo_sapiens.GRCh37.dna.primary_assembly.with_ERCC.GATC.posarray.hdf5"
    ),
    "mapfn": (
        "/data/zfs/references/"
        "human/hg19/mappability/"
        "Homo_sapiens.GRCh37.dna.primary_assembly.with_ERCC.GATC.bowtie2_very_sensitive_N1.readlength_65.counts.pos.hdf5"
    ),
    "blacklistfn": "/data/zfs/references/human/hg19/mappability/ENCFF001TDO.bed.gz",
    "asisi_site_fn": (
        "/data/zfs/deepseq/projects/"
        "DNADamage/KR20160411/data/regions/"
        "Homo_sapiens.GRCh37.dna.primary_assembly.AsiSI_GCGATCGC.bed"
    ),
    # not used any more in favor of TableS2 in Clouaire 2018 Mol Cell
    "breakseq_gh2ax_top_site_fn": (
        "/data/zfs/deepseq/projects/"
        "DNADamage/KR20170201.AsiSI_top_sites/output/"
        "KR20170202.write_breakseq_chipseq_defined_asisi_tophit_subset/"
        "KR20170203.BREAkseq_and_gH2AX_defined_AsiSI_top_sites.tsv"
    ),
}

conf = ChainMap()
if 'SCREPAIR_CONF' in os.environ:
    import yaml
    with open(os.environ['SCREPAIR_CONF'], 'rt') as fh:
        conf.maps.append(yaml.load(fh, Loader=yaml.SafeLoader))
conf.maps.append(DEFAULT_CONF)

# these library types specify which data we can expect per row
DAMID_LIBTYPES = {"DamID", "ChICandDamID", "DamIDandT", "Damaris"}
CHIC_LIBTYPES = {"ChIC", "ChICandDamID"}
CELSEQ_LIBTYPES = {"DamIDandT"}
DAMARIS_LIBTYPES = {"Damaris"}

# TODO: possibly make the paths here configurable?
DAMID_POS_FNFMT = os.path.join(
    conf['PREFIX'],
    "./experiments/{datadir}/data/counts/{limsid}.index{indexnr:02d}.{barcode}.event_counts.pos.hdf5",
)
DAMID_BINNED_FNFMT = os.path.join(
    conf['PREFIX'],
    "./experiments/{datadir}/data/counts/{limsid}.index{indexnr:02d}.{barcode}.event_counts.binsize_{binsize:d}.hdf5",
)
DAMID_BINNED_FNFMT_OLDPOP = os.path.join(
    conf['PREFIX'],
    "./experiments/{datadir}/data/counts/{limsid}.index{indexnr:02d}.{barcode}.counts.binsize_{binsize:d}.hdf5",
)
DAMID_OLDPOPFMT_LIMSIDS = {"KIN1554", "KIN1555", "KIN1583", "KIN1726"}

TX_FNFMT = os.path.join(
    conf['PREFIX'],
    "./experiments/{datadir}/data/counts/{limsid}.index{indexnr:02d}.{barcode}.counts.hdf5",
)

DAMARIS_FNFMT = os.path.join(
    conf['PREFIX'],
    "./experiments/{datadir}/data/counts/{limsid}.index{indexnr:02d}.{barcode}.invalid_pos_reads.counts.hdf5",
)

# try multiple, in order:
CHIC_BINNED_FNFMTS = [
    os.path.join(
        conf['PREFIX'],
        "./experiments/{datadir}/data/counts/{limsid}.index{indexnr:02d}.{barcode}.chic.event_counts.binsize_{binsize:d}.hdf5",
    ),
    os.path.join(
        conf['PREFIX'],
        "./experiments/{datadir}/data/counts/{limsid}.index{indexnr:02d}.{barcode}.chic.counts.binsize_{binsize:d}.hdf5",
    ),
]

BARCODENAMEFMTS = {
    "unspecified": "BC_{barcodenr:03d}",
    "damid": "DamID_BC_{barcodenr:03d}",
    "damid2": "DamID2_BC_{barcodenr:03d}",
    "damid_v2": "BC_DamIDv2_{barcodenr:03d}",
    "damid_v3": "BC_DamIDv3_{barcodenr:03d}",
    "damid_v3_set1": "BCv3set1_BC_{barcodenr:03d}",
    "damid_v3_set2": "BCv3set2_BC_{barcodenr:03d}",
    "chic": "BC_ChIC_{barcodenr:03d}",
}
