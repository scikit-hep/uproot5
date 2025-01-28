# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

from utils import run_test_in_pyodide


# Taken from test_1347_rntuple_floats_suppressed_cols.py
@run_test_in_pyodide(test_file="test_float_types_rntuple_v1-0-0-0.root")
def test_rntuple_with_awkwardforth(selenium):
    import numpy as np

    import uproot

    def truncate_float(value, bits):
        a = np.float32(value).view(np.uint32)
        a &= np.uint32(0xFFFFFFFF) << (32 - bits)
        return a.astype(np.uint32).view(np.float32)

    def quantize_float(value, bits, min, max):
        min = np.float32(min)
        max = np.float32(max)
        if value < min or value > max:
            raise ValueError(f"Value {value} is out of range [{min}, {max}]")
        scaled_value = (value - min) * (2**bits - 1) / (max - min)
        int_value = np.round(scaled_value)
        quantized_float = min + int_value * (max - min) / ((1 << bits) - 1)
        return quantized_float.astype(np.float32)

    with uproot.open("test_float_types_rntuple_v1-0-0-0.root") as f:
        obj = f["ntuple"]

        arrays = obj.arrays()

        min_value = -2.0
        max_value = 3.0

        entry = arrays[0]
        true_value = 1.23456789
        assert entry.trunc10 == truncate_float(true_value, 10)
        assert entry.trunc16 == truncate_float(true_value, 16)
        assert entry.trunc24 == truncate_float(true_value, 24)
        assert entry.trunc31 == truncate_float(true_value, 31)
        assert np.isclose(
            entry.quant1, quantize_float(true_value, 1, min_value, max_value)
        )
        assert np.isclose(
            entry.quant8, quantize_float(true_value, 8, min_value, max_value)
        )
        assert np.isclose(
            entry.quant16, quantize_float(true_value, 16, min_value, max_value)
        )
        assert np.isclose(
            entry.quant20, quantize_float(true_value, 20, min_value, max_value)
        )
        assert np.isclose(
            entry.quant24, quantize_float(true_value, 24, min_value, max_value)
        )
        assert np.isclose(
            entry.quant25, quantize_float(true_value, 25, min_value, max_value)
        )
        assert np.isclose(
            entry.quant32, quantize_float(true_value, 32, min_value, max_value)
        )

        entry = arrays[1]
        true_value = 1.4660155e13
        assert entry.trunc10 == truncate_float(true_value, 10)
        assert entry.trunc16 == truncate_float(true_value, 16)
        assert entry.trunc24 == truncate_float(true_value, 24)
        assert entry.trunc31 == truncate_float(true_value, 31)
        true_value = 1.6666666
        assert np.isclose(
            entry.quant1, quantize_float(true_value, 1, min_value, max_value)
        )
        assert np.isclose(
            entry.quant8, quantize_float(true_value, 8, min_value, max_value)
        )
        assert np.isclose(
            entry.quant16, quantize_float(true_value, 16, min_value, max_value)
        )
        assert np.isclose(
            entry.quant20, quantize_float(true_value, 20, min_value, max_value)
        )
        assert np.isclose(
            entry.quant24, quantize_float(true_value, 24, min_value, max_value)
        )
        assert np.isclose(
            entry.quant25, quantize_float(true_value, 25, min_value, max_value)
        )
        assert np.isclose(
            entry.quant32, quantize_float(true_value, 32, min_value, max_value)
        )

        entry = arrays[2]
        true_value = -6.2875986e-22
        assert entry.trunc10 == truncate_float(true_value, 10)
        assert entry.trunc16 == truncate_float(true_value, 16)
        assert entry.trunc24 == truncate_float(true_value, 24)
        assert entry.trunc31 == truncate_float(true_value, 31)
        assert np.isclose(
            entry.quant1, quantize_float(true_value, 1, min_value, max_value)
        )
        assert np.isclose(
            entry.quant8, quantize_float(true_value, 8, min_value, max_value)
        )
        assert np.isclose(
            entry.quant16, quantize_float(true_value, 16, min_value, max_value)
        )
        assert np.isclose(
            entry.quant20, quantize_float(true_value, 20, min_value, max_value)
        )
        assert np.isclose(
            entry.quant24, quantize_float(true_value, 24, min_value, max_value)
        )
        assert np.isclose(
            entry.quant25,
            quantize_float(true_value, 25, min_value, max_value),
            atol=2e-07,
        )
        assert np.isclose(
            entry.quant32, quantize_float(true_value, 32, min_value, max_value)
        )

        entry = arrays[3]
        true_value = -1.9060668
        assert entry.trunc10 == truncate_float(true_value, 10)
        assert entry.trunc16 == truncate_float(true_value, 16)
        assert entry.trunc24 == truncate_float(true_value, 24)
        assert entry.trunc31 == truncate_float(true_value, 31)
        assert np.isclose(
            entry.quant1, quantize_float(true_value, 1, min_value, max_value)
        )
        assert np.isclose(
            entry.quant8, quantize_float(true_value, 8, min_value, max_value)
        )
        assert np.isclose(
            entry.quant16, quantize_float(true_value, 16, min_value, max_value)
        )
        assert np.isclose(
            entry.quant20, quantize_float(true_value, 20, min_value, max_value)
        )
        assert np.isclose(
            entry.quant24, quantize_float(true_value, 24, min_value, max_value)
        )
        assert np.isclose(
            entry.quant25, quantize_float(true_value, 25, min_value, max_value)
        )
        assert np.isclose(
            entry.quant32, quantize_float(true_value, 32, min_value, max_value)
        )
