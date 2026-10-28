import importlib.util
from pathlib import Path
import struct

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    'graph_dump_compare', Path(__file__).resolve().parents[1] / 'scripts/graph-dump-compare.py')
assert spec is not None and spec.loader is not None
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def record(file, size):
    return compare.Record(fd=file.fileno(), offset=4, n_bytes=size, name='result_output', type=compare.GGML_TYPE_F32)


def test_short_reads(tmp_path, monkeypatch):
    expected = np.array([1, 2, 3, 4], dtype='<f4')
    path = tmp_path / 'tensor.bin'
    path.write_bytes(b'HEAD' + expected.tobytes())
    pread = compare.os.pread
    calls = []

    def short_read(fd, size, offset):
        calls.append((size, offset))
        return pread(fd, min(size, 3), offset)

    monkeypatch.setattr(compare.os, 'pread', short_read)
    with path.open('rb') as file:
        np.testing.assert_array_equal(record(file, expected.nbytes)['data'], expected)
    assert len(calls) > 1


def test_unexpected_eof(tmp_path):
    path = tmp_path / 'tensor.bin'
    path.write_bytes(b'HEAD' + np.array([1, 2, 3], dtype='<f4').tobytes())
    with path.open('rb') as file, pytest.raises(EOFError, match='expected 16 bytes, read 12'):
        record(file, 16)['data']


def test_bounded_reads(tmp_path, monkeypatch):
    size = 9 * 1024 * 1024
    path = tmp_path / 'tensor.bin'
    path.write_bytes(b'HEAD' + bytes(size))
    pread = compare.os.pread
    sizes = []

    def read(fd, count, offset):
        sizes.append(count)
        return pread(fd, count, offset)

    monkeypatch.setattr(compare.os, 'pread', read)
    with path.open('rb') as file:
        assert record(file, size)['data'].nbytes == size
    assert sizes == [8 * 1024 * 1024, 1024 * 1024]


def test_truncated_dump(tmp_path):
    path = tmp_path / 'truncated.bin'
    header = b'LGDUMP\0\1' + struct.pack('<IIIi', 1, 1, 1, 0)
    record_header = struct.pack('<I', 1) + b'x' + struct.pack('<II4qQ', compare.GGML_TYPE_F32, 0, 4, 1, 1, 1, 1 << 40)
    path.write_bytes(header + record_header + bytes(4))
    with pytest.raises(EOFError, match='truncated tensor x'):
        compare.load(path)
