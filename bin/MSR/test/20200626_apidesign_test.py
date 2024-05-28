import numpy as np


# some super cheesy input data
fg = np.zeros(100, dtype=int)
fg[10:20] = 20
fg[60:65] = 10
fg[80:90] = 25

bg = np.ones(100, dtype=int)
bg *= 10

# add pseudo-chrom
fg = {'1': fg}
bg = {'1': bg}

# calculate scaling factor (simply based on totals)
chroms = {"1", }
fg_total = sum(fg[chrom].sum() for chrom in chroms)
bg_total = sum(bg[chrom].sum() for chrom in chroms)

sf = fg_total / bg_total


from msr.msr import MSR


def test_api():
    m = MSR(
        fg=fg,
        bg=bg,
        scaling_factor=sf,
    )
    mr = m.get_results()
