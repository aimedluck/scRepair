import pickle
import gzip
import struct
import warnings

import numpy as np
import scipy.signal
from scipy.interpolate import interp1d


def level2stddev(l):
    return np.exp2((l - 2) / 2)


def discrete_gaussian(stddev):
    #width = int(np.ceil(stddev * 3)) * 2 + 1  # ensure odd; take <1% error (on sides) for granted
    width = int(np.ceil(stddev * 6)) * 2 + 1

    return scipy.signal.windows.gaussian(width, stddev)


def get_scale_space(v, stddev):
    return scipy.signal.convolve(
        v.astype(float),
        discrete_gaussian(stddev),
        mode='same',
    )


def get_breaks(v):
    """
    Returns segment starts (usable with ufunc reduceat)
    """
    return np.where(np.r_[True, v[1:] != v[:-1]])[0]


class SegmentMap:
    def __init__(self):
        self._layers = []

    def append(self, item):
        if not isinstance(item, SegmentLayer):
            raise ValueError("Can only append SegmentLayer")

        return self._layers.append(item)

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, i):
        if not isinstance(i, int):
            raise ValueError()

        return self._layers[i]

    def segment_counts(self):
        return [len(self._layers[i]) for i in range(len(self))]

    def serialize(self):
        serialized_layers = [
            l.serialize()
            for l in self._layers
        ]

        payload = pickle.dumps(serialized_layers)
        return payload

    @classmethod
    def deserialize(cls, payload):
        layer_payloads = pickle.loads(payload)
        layers = [
            SegmentLayer.deserialize(l)
            for l in layer_payloads
        ]

        obj = cls()
        obj._layers = layers

        return obj

    def to_dense(self, attr=None, sign=True):
        return np.array([
            layer.to_dense(attr=attr, sign=sign)
            for layer in self._layers
        ])


# this should be part of numpy ... >:/
def serialize_numpy_array(a):
    assert a.ndim < 256
    prefix = struct.pack(">cB", a.dtype.char.encode("ascii"), a.ndim)
    dims = struct.pack(">%dI" % a.ndim, *a.shape)
    return prefix + dims + a.tobytes()


def deserialize_numpy_array(payload):
    char, ndim = struct.unpack_from(">cB", payload)
    char = char.decode("ascii")

    payload_prefix_len = struct.calcsize(">cB")
    shape_prefix_len = struct.calcsize(">%dI" % ndim)

    shape = struct.unpack_from(">%dI" % ndim, payload, offset=payload_prefix_len)

    a = np.frombuffer(payload, dtype=char, offset=payload_prefix_len + shape_prefix_len)

    return a.reshape(shape)


class SegmentLayer:
    DATAKEYS = {
        "start": "uint64",
        "size": "uint64",
        "isign": "uint8",
        "sfc_score": "float64",
        "fc": "float64",
    }

    def __init__(self, data):
        if not self.validate(data):
            raise ValueError("Invalid data")

        self.data = data

    def validate(self, data):
        if not all(key in data for key in self.DATAKEYS):
            return False

        nsegments = len(data['start'])
        for key in data:
            if len(data[key]) != nsegments:
                return False

            if data[key].ndim != 1:
                return False

        return True

    def __len__(self):
        return len(self.data["start"])

    def serialize(self, compresslevel=4):
        compressed_data = {}
        for key in self.data:
            compressed_data[key] = gzip.compress(
                serialize_numpy_array(self.data[key]),
                compresslevel=compresslevel,
            )

        payload = pickle.dumps(compressed_data)
        return payload

    @classmethod
    def deserialize(cls, payload):
        compressed_data = pickle.loads(payload)
        data = {key: deserialize_numpy_array(gzip.decompress(value)) for (key, value) in compressed_data.items()}
        return cls(data)

    def to_dense(self, attr=None, sign=True):
        if attr is None:
            attr = "sfc_score"
        else:
            if attr not in self.data:
                raise ValueError("Unknown attribute: %s" % attr)

        if sign:
            return np.repeat(
                np.multiply(
                    self.data[attr],
                    np.sign(self.data['isign'].astype(int) - 1),
                ),
                self.data["size"].astype(int),
            )
        else:
            return np.repeat(
                self.data[attr],
                self.data["size"].astype(int),
            )


class MSR:
    # helper vectors (round away from "edges" of conf. int.)
    CI_ADD = np.array([-1, 1])
    OFFSET_ADD = np.array([0, 1])

    # log10 scale; implicitly determines nr of gridpoints
    INTERP_XMIN = -4
    INTERP_XMAX = 9
    INTERP_XDELTA = 0.1

    def __init__(self, fg, bg, scaling_factor, max_stddev_bins=None, mapab=None, segmentation_pvalue=5e-2, enrichment_pvalue=1e-5, vmr=1.0):
        """
        fg, bg, mapab: dict of chromosome to one-dimensional vector.
        Each key of `fg` must be present in `bg` and `mapab`.
        All vectors must be of the same shape.
        `fg` vectors must be of integer type (discrete count data).
        """
        chroms = set(fg.keys())
        assert chroms.issubset(bg.keys())
        if mapab is not None:
            assert chroms.issubset(mapab.keys())
        else:
            mapab = {chrom: np.ones(fg[chrom].shape, dtype="bool") for chrom in chroms}

        for chrom in chroms:
            assert np.issubdtype(fg[chrom].dtype, np.integer)
            assert mapab[chrom].dtype.kind == 'b'
            assert fg[chrom].ndim == bg[chrom].ndim == mapab[chrom].ndim == 1

            assert fg[chrom].shape == bg[chrom].shape == mapab[chrom].shape

        self.fg = fg
        self.bg = bg
        self.mapab = mapab

        assert scaling_factor > 0.0
        self.scaling_factor = scaling_factor

        self.vmr = float(vmr)
        assert self.vmr >= 1.0, "0 < VMR < 1. will probably lead to unexpected behaviour but might be implemented in the future"
        assert self.vmr > 0., "VMR < 0. is invalid"

        # check for sensible scaling factor; warn if possibly wrong
        # TODO de-gate
        if __debug__:
            fgsum = sum(fg[chrom].sum() for chrom in chroms)
            bgsum = sum(bg[chrom].sum() for chrom in chroms)
            naive_scaling_factor = fgsum / bgsum
            if np.abs(np.log(naive_scaling_factor / scaling_factor)) > np.log(100):
                warnings.warn(
                    "Possible error in scaling factor? (%.1f vs %.1f)" % (scaling_factor, naive_scaling_factor)
                )

        assert 0 < segmentation_pvalue < 1
        assert 0 < enrichment_pvalue < 1
        self.segmentation_pvalue = segmentation_pvalue
        self.enrichment_pvalue = enrichment_pvalue
        # create intervals
        self.segmentation_pval_iv = np.array([segmentation_pvalue / 2, 1 - segmentation_pvalue / 2])
        self.enrichment_pval_iv = np.array([enrichment_pvalue / 2, 1 - enrichment_pvalue / 2])

        self.max_stddev_bins = int(max_stddev_bins)
        assert self.max_stddev_bins > 0

        # TODO add option
        self.use_fast_ppf_interpolation = True
        if self.use_fast_ppf_interpolation:
            self._setup_ppf_interpolation()

        self._results = None

        return

    def _run_msr(self):
        self._results = {}
        chroms = self.fg.keys()
        for chrom in chroms:
            segment_map = self._segment_and_score_chrom(chrom)

            self._results[chrom] = segment_map

        return

    def _segment_and_score_chrom(self, chrom):
        # (calculate nlevels)
        nbins = len(self.fg[chrom])
        nlevels = min(
            map(
                lambda x: int(np.floor(np.log2(x) * 2 + 2)),
                [self.max_stddev_bins, nbins / 3]
            )
        )

        mapab = self.mapab[chrom].astype(int)

        segment_map = SegmentMap()

        for level in range(nlevels):
            # segment
            # - get scale spaces
            # - calc CI around fg
            # - determine bg crossings of CI
            stddev_bins = level2stddev(level)

            # run segmentation
            segment_start, segment_ci_value, segment_size = self._perform_segmentation(chrom, level)

            # filter segments
            # calculate number of mappable bins per segment
            bins_mappable = np.add.reduceat(mapab, segment_start)
            # calculate fraction of segment mappable
            frac_mappable = bins_mappable / segment_size
            assert not np.isnan(frac_mappable).any()

            # segments are only eligible if they are
            # - large enough (> 2x stddev at this level)
            # - not within CI (ci_value == 1)
            # - "mappable enough"  # TODO make this configurable
            w_segment_eligible = (
                (segment_ci_value != 1)
                & (segment_size > (2 * stddev_bins))
                & (frac_mappable >= 0.5)
            )

            # score segments
            seg_sfc_score, seg_fc, fg_seg, bg_seg, fg_seg_enr_ci, bg_exp = self._perform_segment_scoring(
                chrom,
                segment_start,
                w_segment_eligible,
                segment_ci_value,
            )

            layer = SegmentLayer({
                "start": segment_start,
                "size": segment_size,
                "isign": segment_ci_value,
                "sfc_score": seg_sfc_score,
                "fc": seg_fc,
                "fg_seg": fg_seg,
                "bg_seg": bg_seg,
                "fg_enr_ci_low": fg_seg_enr_ci[:, 0].copy(),
                "fg_enr_ci_high": fg_seg_enr_ci[:, 1].copy(),
                "bg_exp": bg_exp,
            })
            segment_map.append(layer)

        assert len(segment_map) == nlevels

        return segment_map

    def _perform_segmentation(self, chrom, level):
        fg = self.fg[chrom]
        bg = self.bg[chrom]

        if level == 0:
            fg_ss = fg.astype(float)
            bg_ss = bg.astype(float)
        else:
            stddev_bins = level2stddev(level)
            # NOTE profiling shows that >90% of MSR running time (obtaining
            # segmentation map) is spent on calculating the scale spaces
            # (binsize 1kb, overhead might be different for larger binsize, but
            # then again, running on all chroms is ~200ms with 100kb binsize so
            # not really relevant)
            fg_ss = np.maximum(0., get_scale_space(fg, stddev_bins))
            bg_ss = np.maximum(0., get_scale_space(bg, stddev_bins))

        # establish CI around fg
        ci_fg = self.get_ci(fg_ss)
        assert (ci_fg[:, 1] > ci_fg[:, 0]).all()
        # check whether bg falls under/within/above CI
        # TODO: CI_ADD necessary here?
        ci_value = (
            np.expand_dims(bg_ss * self.scaling_factor, -1) < ci_fg
        ).sum(axis=1)
        # segment
        segment_start = get_breaks(ci_value)
        segment_ci_value = ci_value[segment_start]

        # calculate sizes
        segment_size = np.empty_like(segment_start)
        np.subtract(segment_start[1:], segment_start[:-1], out=segment_size[:-1])
        segment_size[-1] = len(fg) - segment_start[-1]
        assert (segment_size > 0).all()

        return segment_start, segment_ci_value, segment_size

    def _perform_segment_scoring(self, chrom, segment_starts, w_segment_eligible, segment_ci_value):
        # - sum raw counts within segments
        # - calculate CI on fg
        # - calculate SFCs
        # - calculate raw FCs
        fg = self.fg[chrom]
        bg = self.bg[chrom]

        # sum binned counts per segment
        fg_seg = np.add.reduceat(fg, segment_starts)
        bg_seg = np.add.reduceat(bg, segment_starts)

        # calculate CI on fg segments
        # NB: not using interpolation because the nr of segments is assumed
        # to be small
        assert len(segment_starts) <= 100_000, "assuming the nr of segments is small"
        # TODO: CI_ADD necessary here?
        # TODO: wouldnt Poisson (or NB) be more applicable here since we
        # are working with discrete counts?
        fg_seg_enr_ci = np.nan_to_num(scipy.stats.gamma.ppf(
            self.enrichment_pval_iv,
            np.maximum(0, np.expand_dims(fg_seg, -1) + self.CI_ADD) / self.vmr,
            scale=self.vmr,
        ))
        # TODO: CI_ADD necessary here?
        fg_seg_value = (
            (np.expand_dims(bg_seg * self.scaling_factor, -1) + self.CI_ADD) < fg_seg_enr_ci
        ).sum(axis=1)
        # exclude the segments that were excluded earlier
        fg_seg_value[~w_segment_eligible] = 1
        # assume that enrichments here were also enrichments in the scale
        # space
        # (and idem for depletions)
        w_seg_enr = (fg_seg_value == 2)
        w_seg_dep = (fg_seg_value == 0)

        # these assumptions seem to not always hold...
        # see e.g. KIN1554.index07.BC_037 (a D5 sample, not really important)
        #assert (segment_ci_value[w_seg_enr] == 2).all()
        #assert (segment_ci_value[w_seg_dep] == 0).all()

        # calculate SFCs (for enrichments and depletions separately)
        bg_exp = np.maximum(0.5, bg_seg * self.scaling_factor)
        seg_sfc_scores = np.zeros(len(segment_starts), dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            seg_sfc_scores[w_seg_enr] = np.maximum(0., np.log2(
                fg_seg_enr_ci[:, 0][w_seg_enr] / bg_exp[w_seg_enr]
            ))
            seg_sfc_scores[w_seg_dep] = np.maximum(0., np.log2(
                bg_exp[w_seg_dep] / fg_seg_enr_ci[:, 1][w_seg_dep]
            ))

        assert (seg_sfc_scores >= 0).all()

        # calculate raw log2FCs
        seg_fcs = np.log2(np.maximum(0.5, fg_seg) / bg_exp)

        return seg_sfc_scores, seg_fcs, fg_seg, bg_seg, fg_seg_enr_ci, bg_exp

    def get_results(self):
        if self._results is None:
            self._run_msr()

        return self._results  # TODO what kind of results are a good API?

    def get_ci(self, x):
        assert x.ndim == 1

        if not self.use_fast_ppf_interpolation:
            # TODO: CI_ADD necessary here?
            ci = np.nan_to_num(scipy.stats.gamma.ppf(
                self.segmentation_pval_iv,
                np.maximum(0, np.expand_dims(x, -1) + self.CI_ADD) / self.vmr,
                scale=self.vmr,
            ))
        else:
            # use interpolation
            # assume it has been properly initialized
            with np.errstate(divide='ignore', invalid='ignore'):
                xtransformed = np.maximum(self.INTERP_XMIN, np.log10(x))
            ci = np.array([
                self._ci_interp_low(xtransformed),
                self._ci_interp_high(xtransformed),
            ]).T

        return ci

    @staticmethod
    def _ci_interp_transform(x):
        return np.log10(x)

    def _setup_ppf_interpolation(self):
        # determine extrema (log10 scale)
        xpoints = np.arange(
            self.INTERP_XMIN,
            self.INTERP_XMAX + self.INTERP_XDELTA,
            self.INTERP_XDELTA,
        )

        # define CI at grid
        # TODO: CI_ADD necessary here?
        ci_lut = np.nan_to_num(scipy.stats.gamma.ppf(
            self.segmentation_pval_iv,
            np.maximum(0, np.expand_dims(np.power(10., xpoints), -1) + self.CI_ADD) / self.vmr,
            scale=self.vmr,
        ))

        # store the xy grid as interpolation
        self._ci_interp_low = interp1d(xpoints, ci_lut[:, 0], assume_sorted=True)
        self._ci_interp_high = interp1d(xpoints, ci_lut[:, 1], assume_sorted=True)

        return
