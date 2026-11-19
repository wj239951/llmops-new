# 图片输入的纯函数测试（不依赖 OCR 引擎与 GPU）。
import base64

from app.services.image_input import decode_image, merge_query, public_meta, tidy_text


def test_merge_query_order():
    assert merge_query("型号 X1", "这个型号怎么保养") == "型号 X1\n\n这个型号怎么保养"
    assert merge_query("", "只有问题") == "只有问题"
    assert merge_query("只有图片", "") == "只有图片"
    assert merge_query("", "") == ""


def test_tidy_text_collapses_and_dedupes():
    raw = "  设备名称   注塑机  \n\n设备名称   注塑机\n型号 ZX-1200\n"
    assert tidy_text(raw) == "设备名称 注塑机\n型号 ZX-1200"


def test_decode_image_data_uri_and_bad_input():
    payload = b"\x89PNG\r\n\x1a\nfake"
    uri = "data:image/png;base64," + base64.b64encode(payload).decode()
    assert decode_image(uri) == payload
    assert decode_image("!!!not-base64!!!") in (None, b"")
    assert decode_image("") is None


def test_public_meta_hides_unapplied():
    assert public_meta(None) is None
    assert public_meta({"applied": False}) is None
    meta = {"applied": True, "engine": "rapidocr", "chars": 6, "ms": 12,
            "truncated": False, "text": "型号 X1"}
    out = public_meta(meta)
    assert out["engine"] == "rapidocr" and out["text"] == "型号 X1"
