# 领域同义词归一：把口语说法映射成语料里实际使用的规范术语（只作用于检索）。
from __future__ import annotations

# 口语/同义说法 -> 规范术语（以知识库里的实际写法为准）
ALIASES: dict[str, str] = {
    "密闭车间": "受限空间",
    "密闭空间": "受限空间",
    "密闭区域": "受限空间",
    "有限空间": "受限空间",
    "受限区域": "受限空间",
    "受限作业": "受限空间作业",
    "空压机房": "空压站",
    "压缩机": "空压机",
}


def normalize_terms(text: str) -> tuple[str, list[str]]:
    """把口语说法替换成规范术语，返回 (归一后的文本, 命中的替换说明)。

    只用于检索查询；给模型看的提问与最终回答仍用用户原话。
    """
    if not text:
        return "", []
    out = text
    hits: list[str] = []
    # 长词优先，避免「密闭空间」被「密闭」类短词抢先替换
    for src in sorted(ALIASES, key=len, reverse=True):
        dst = ALIASES[src]
        if src != dst and src in out:
            out = out.replace(src, dst)
            hits.append(f"{src}→{dst}")
    return out, hits
