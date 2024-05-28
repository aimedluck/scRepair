import gzip
import base64

import numpy as np
import scipy.signal


from msr.msr import MSR
from msr.segmentgraph import segment_map_to_graph, prune_segment_graph, flatten_segment_graph, validate_single_node_per_position, flat_score

import pytest


def np2buf(v):
    return base64.b64encode(
        gzip.compress(
            v.tobytes(),
        )
    )


def buf2np(b, dtype):
    return np.frombuffer(
        gzip.decompress(
            base64.b64decode(b)
        ),
        dtype=dtype,
    )


@pytest.fixture
def setup_msr_testdata1():
    BG_DATA = b'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'

    FG_DATA = b'H4sIAAQd6V4C/+2c0RbDEBBEM6v9/1/uux5dEcpy58WhkrBm14jKdcWELgC+YU65snwqlFvhulk83JXvpX6lm/VVyJ9mt9HPsc52Jo6vyQc54/vv59VeVzsfqPK+tgi/NZk/ullvdz9R0P4piH1zvLLUnLhhjXyuzedpatSltfGn1R9bx6fXvK5N/X9XnTO7nXbFBDr2DDvpZurFQ+vM+1N4aA/r9dK7T8dHD/sHP4iXxGu/vYIH+M1B86KcdXS+7/EurGdT9rt1tq+C2bW0j6Tgdhj9fpP4BUDcdW6p3IuL78J8kx62azddQhwFjDsAAAAQWycD9NYvrL7Pvct7LfQzAAB9EmPeAWvNb1qUx7PsiY7ATyK3G/6ewetR+0nm5PWn/ozqrxbz29b/C9eeHzk9Tue6onS+aHS7o+lyBR3v03VEFHvY5v4DL8GKPPF0lff9h9pz/L2+J6DJ9gFrrmsA/B0RV6La7wPjfJ5N6E0AAA=='

    MAPAB_DATA = b'H4sIAGkd6V4C/2NgAANGTMCAk0NdwMCA0wl4HDQAYKDtHwWjgNZJnIGRgWEI5ApogcEwHAEjnEARhYc1enGJrhqmZjQ1jwJSMxXj0Eo2DCO1VmaAFNUAGwAeXr0JAAA='

    chrom = '1'
    bg_total = 2824250
    sf = 1342.1177408124581

    # load data into numpy arrays:
    w_mapab = buf2np(MAPAB_DATA, dtype="bool")
    fg = buf2np(FG_DATA, dtype=int)
    bg = buf2np(BG_DATA, dtype=int)
    assert bg.shape == fg.shape == w_mapab.shape == (2493, )

    # run MSR
    m = MSR(
        {chrom: fg},
        {chrom: bg},
        scaling_factor=sf / bg_total,
        mapab={chrom: w_mapab},
        segmentation_pvalue=2 * 5e-2,
        enrichment_pvalue=2 * 1e-5,
    )

    return m


def test_msr_1(setup_msr_testdata1):
    chrom = '1'
    m = setup_msr_testdata1
    mr = m.get_results()
    segment_map = mr[chrom]
    nlevels = len(segment_map)

    Y = np.array([
        np.repeat(
            segment_map[level].data['sfc_score'] * np.sign(segment_map[level].data['isign'].astype(int) - 1), segment_map[level].data["size"].astype(int))
        for level in range(nlevels)
    ])

    Y_DATA = b'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'

    Ytrue = buf2np(Y_DATA, float).reshape(-1, 2493)

    assert Y.shape == Ytrue.shape

    assert np.isclose(Y, Ytrue, rtol=0.2, atol=1e-3).all()


def test_msr_negativecontrol_1():
    """
    Test a "flat" sample where nothing should be significant
    """

    L = 1000
    fg = np.ones(L, dtype=int)
    bg = np.ones(L)
    mapab = np.ones(L, dtype=bool)

    chromname = "mychrom"  # whatever
    m = MSR(
        fg={chromname: fg},
        bg={chromname: bg},
        mapab={chromname: mapab},
        scaling_factor=1.,
    )

    mr = m.get_results()
    segment_map = mr[chromname]
    assert len(segment_map) == int(np.ceil(np.log2(L * 2)) + 2)

    segment_graph = segment_map_to_graph(segment_map)
    segment_graph_pruned = segment_graph.copy()
    prune_segment_graph(segment_graph_pruned, segment_map.segment_counts(), T=1.05)

    sv = flatten_segment_graph(segment_graph_pruned)
    # validate stepvector
    validate_single_node_per_position(sv)
    # calculate score vector
    score = flat_score(segment_map, sv, l=L)
    assert (score == 0.).all()


def test_msr_positivecontrol_1():
    L = 1000
    fg = np.ones(L, dtype=int)
    bg = np.ones(L)
    mapab = np.ones(L, dtype=bool)

    w_signal = np.zeros(L, dtype=bool)
    w_signal[100:120] = True
    fg[w_signal] *= 10

    chromname = "mychrom"  # whatever
    m = MSR(
        fg={chromname: fg},
        bg={chromname: bg},
        mapab={chromname: mapab},
        scaling_factor=1.,
    )

    mr = m.get_results()
    segment_map = mr[chromname]
    assert len(segment_map) == int(np.ceil(np.log2(L * 2)) + 2)

    segment_graph = segment_map_to_graph(segment_map)
    segment_graph_pruned = segment_graph.copy()
    prune_segment_graph(segment_graph_pruned, segment_map.segment_counts(), T=1.05)

    sv = flatten_segment_graph(segment_graph_pruned)
    # validate stepvector
    validate_single_node_per_position(sv)
    # calculate score vector
    score = flat_score(segment_map, sv, l=L)
    assert (score[w_signal] > 0.).any()
    # apply a bit of smoothing since a few bins around the fake signal will also be regarded as enriched
    assert (
        score[
            scipy.signal.convolve(w_signal.astype(float), np.ones(10), mode='same') == 0
        ] == 0.
    ).all()


def test_msr_negativecontrol_2():
    """
    peak but insignificant
    """
    L = 1000
    fg = np.ones(L, dtype=int)
    bg = np.ones(L)
    mapab = np.ones(L, dtype=bool)

    w_signal = np.zeros(L, dtype=bool)
    w_signal[100:101] = True
    fg[w_signal] *= 2

    chromname = "mychrom"  # whatever
    m = MSR(
        fg={chromname: fg},
        bg={chromname: bg},
        mapab={chromname: mapab},
        scaling_factor=1.,
        # sensitive parameters
        enrichment_pvalue=0.1,
        segmentation_pvalue=0.49,
    )

    mr = m.get_results()
    segment_map = mr[chromname]
    assert len(segment_map) == int(np.ceil(np.log2(L * 2)) + 2)

    segment_graph = segment_map_to_graph(segment_map)
    segment_graph_pruned = segment_graph.copy()
    prune_segment_graph(segment_graph_pruned, segment_map.segment_counts(), T=1.05)

    sv = flatten_segment_graph(segment_graph_pruned)
    # validate stepvector
    validate_single_node_per_position(sv)
    # calculate score vector
    score = flat_score(segment_map, sv, l=L)
    assert (score[w_signal] == 0.).all()


def test_msr_negativecontrol_3():
    """
    peak but very low signal in fg (insignificant)
    """
    L = 1000
    fg = np.ones(L, dtype=int)
    bg = np.ones(L)
    mapab = np.ones(L, dtype=bool)

    w_signal = np.zeros(L, dtype=bool)
    w_signal[100:101] = True
    fg[w_signal] *= 1
    fg[~w_signal] *= 0

    chromname = "mychrom"  # whatever
    m = MSR(
        fg={chromname: fg},
        bg={chromname: bg},
        mapab={chromname: mapab},
        # scaling factor indicating high IC
        scaling_factor=1e-4,
        # sensitive parameters
        enrichment_pvalue=5e-2,
        segmentation_pvalue=0.45,
    )

    mr = m.get_results()
    segment_map = mr[chromname]
    assert len(segment_map) == int(np.ceil(np.log2(L * 2)) + 2)

    segment_graph = segment_map_to_graph(segment_map)
    segment_graph_pruned = segment_graph.copy()
    prune_segment_graph(segment_graph_pruned, segment_map.segment_counts(), T=1.05)

    sv = flatten_segment_graph(segment_graph_pruned)
    # validate stepvector
    validate_single_node_per_position(sv)
    # calculate score vector
    score = flat_score(segment_map, sv, l=L, attr="sfc_score")
    assert (score[w_signal] == 0.).all()
