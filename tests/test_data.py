"""Tests for Phase 0 data extraction and label unification."""

from dental_model.data.extract import load_data_config
from dental_model.data.unify_labels import (
    ROBOFLOW_CLASS_MAP,
    compute_sha256,
    generate_checksums,
    remap_yolo_label_file,
)


def test_load_data_config():
    cfg = load_data_config("configs/data_paths.yaml")
    assert "sources" in cfg
    assert "caries_spectra" in cfg["sources"]
    assert "roboflow_detection" in cfg["sources"]


def test_remap_yolo_label_file(tmp_path):
    src_lbl = tmp_path / "sample.txt"
    dst_lbl = tmp_path / "remapped.txt"

    lines = [
        "0 0.5 0.5 0.2 0.2\n",
        "1 0.4 0.4 0.1 0.1\n",
        "2 0.3 0.3 0.1 0.1\n",  # filtered
        "4 0.6 0.6 0.2 0.2\n",
        "5 0.7 0.7 0.3 0.3\n",
    ]
    src_lbl.write_text("".join(lines), encoding="utf-8")

    count = remap_yolo_label_file(src_lbl, dst_lbl, ROBOFLOW_CLASS_MAP)
    assert count == 4

    out_lines = dst_lbl.read_text(encoding="utf-8").splitlines()
    assert len(out_lines) == 4
    assert out_lines[0].startswith("2 ")
    assert out_lines[1].startswith("2 ")
    assert out_lines[2].startswith("0 ")
    assert out_lines[3].startswith("1 ")


def test_compute_sha256(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello dental model", encoding="utf-8")
    h = compute_sha256(f)
    assert isinstance(h, str)
    assert len(h) == 64


def test_generate_checksums(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "labels.csv").write_text("image_path,label\n1.jpg,healthy\n", encoding="utf-8")
    out_json = processed / "checksums.json"

    sums = generate_checksums(processed, out_json)
    assert out_json.exists()
    assert len(sums) == 1
