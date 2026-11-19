# 领域同义词归一的单元测试。
from app.services.query_alias import ALIASES, normalize_terms


def test_confined_space_aliases():
    out, hits = normalize_terms("在密闭车间作业需要注意什么")
    assert out == "在受限空间作业需要注意什么"
    assert hits == ["密闭车间→受限空间"]

    out, hits = normalize_terms("有限空间的氧含量要求")
    assert "受限空间" in out and "有限空间" not in out
    assert hits


def test_no_alias_leaves_text_untouched():
    q = "注塑机 ZX-1200 巡检周期"
    assert normalize_terms(q) == (q, [])


def test_longest_match_first():
    # 「受限区域」比「受限」长，且不应产生自我替换
    out, _ = normalize_terms("受限区域的检测要求")
    assert "受限区域" not in out


def test_empty_input():
    assert normalize_terms("") == ("", [])


def test_alias_targets_exist_in_corpus_terms():
    # 目标词本身不应再被别的规则改写（避免链式替换）
    for src, dst in ALIASES.items():
        assert ALIASES.get(dst, dst) == dst
