#!/usr/bin/env python3
"""
Extract all figure/table drawing requirements from 7 DOCX files under 2026-02-07.
Output:
- output/doc/2026-02-07_figures/docs/plot_requirements_structured.json
- output/doc/2026-02-07_figures/docs/plot_requirements_structured.md
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from docx import Document


ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = ROOT / "2026-02-07"
OUT_DIR = ROOT / "output" / "doc" / "2026-02-07_figures" / "docs"
OUT_JSON = OUT_DIR / "plot_requirements_structured.json"
OUT_MD = OUT_DIR / "plot_requirements_structured.md"

REQ_PATTERN = re.compile(r"([图表])\s*([0-9M]+(?:[-－—][0-9A-Za-zXx]+))")
FLOWCHART_KEYWORDS = ("流程图", "架构图", "数据流图", "路线图", "框架图", "结构图", "闭环架构")
HEATMAP_KEYWORDS = ("热力图", "风险图", "参数场", "可视化", "预演")
CURVE_KEYWORDS = ("曲线", "沉陷", "滑移", "牵引力", "对比")
IMAGE_PANEL_KEYWORDS = ("分割结果", "重建结果", "示例", "照片")

DOC_KEY_PATTERNS = {
    "outline": "170页",
    "ch1": "第1章",
    "ch2": "第2章",
    "ch3": "第3章",
    "ch4": "第4章",
    "ch5": "第5章",
    "ch6": "第6章",
}

TITLE_OVERRIDES: Dict[str, str] = {
    "图M-1": "跨章节接口契约总览",
    "图M-2": "任务预演风险热力图",
    "图1-1": "月面巡视器主要风险源分类示意图",
    "图1-2": "现有技术体系脉络图",
    "图1-3": "科学问题逻辑关系图",
    "图1-4": "论文章节结构与技术路线图",
    "图2-1": "UQ-DT理论框架图",
    "图2-2": "接口数据契约矩阵图",
    "图2-X": "第二章接口总图（抽象编号）",
    "图3-1": "三层架构数据流图",
    "图3-2": "AdaScale-GSFR算法流程图",
    "图3-3": "语义分割结果示例",
    "图3-4": "物理参数场可视化热力图",
    "图3-5": "1/6g与1g沉陷曲线对比",
    "图4-1": "轮-壤接触力学示意图",
    "图4-2": "1/6g与1g沉陷曲线对比",
    "图4-3": "牵引力-滑移率曲线",
    "图4-4": "RLS在线辨识算法流程图",
    "图4-5": "数字孪生闭环架构图",
    "图4-6": "Apollo LRV轮迹对比验证图",
    "图5-1": "SiaT-Hough网络架构图",
    "图5-2": "改进ORB-SLAM2系统流程图",
    "图5-3": "数字孪生环境下避障预演图",
    "图5-4": "岩石识别与重建结果图",
    "图6-1": "基于增强五维模型的分层路径规划框架图",
    "图6-2": "A*-D3QN-Opt算法架构图",
    "图6-3": "全局任务预演风险热力图",
    "图6-4": "五维奖励函数权重敏感性分析图",
    "图6-5": "通讯时延补偿效果对比图",
    "表3-1": "Apollo力学参数先验分布表",
    "表3-5": "D3QN/D3QN-PER/A-D3QN-Opt性能对比表",
    "表4-1": "Apollo验证数据对比表",
    "表5-1": "SiaT-Hough与其他方法性能对比表",
    "表5-2": "避障策略风险评估参数表",
    "表6-1": "不同算法在三种复杂度环境下性能对比表",
}

DEFERRED_FLOW_IDS = {
    "图M-1",
    "图1-2",
    "图1-3",
    "图1-4",
    "图2-1",
    "图3-1",
    "图3-2",
    "图4-4",
    "图4-5",
    "图5-1",
    "图5-2",
    "图6-1",
    "图6-2",
}

DATA_PRIORITY_IDS = {
    "图M-2",
    "图2-2",
    "图3-3",
    "图3-4",
    "图3-5",
    "图4-1",
    "图4-2",
    "图4-3",
    "图4-6",
    "图5-3",
    "图5-4",
    "图6-3",
    "图6-4",
    "图6-5",
    "表3-1",
    "表3-5",
    "表4-1",
    "表5-1",
    "表5-2",
    "表6-1",
}


@dataclass
class Requirement:
    doc_key: str
    doc_name: str
    req_id: str
    req_type: str
    title: str
    excerpt: str
    chart_kind: str
    visual_elements: List[str] = field(default_factory=list)
    style_requirements: List[str] = field(default_factory=list)
    size_specs: List[str] = field(default_factory=list)
    special_notes: List[str] = field(default_factory=list)
    flowchart_deferred: bool = False
    data_priority: bool = False
    expected_data: List[str] = field(default_factory=list)
    covered_by: List[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text.strip())
    return text


def norm_dash(text: str) -> str:
    return text.replace("—", "-").replace("－", "-")


def canonical_doc_key(req_id: str, fallback_doc_key: str) -> str:
    # 图M-x / 表M-x -> outline
    body = req_id[1:]
    if body.upper().startswith("M-"):
        return "outline"
    # 图2-X 也归第2章
    m = re.match(r"^([0-9]+)-", body)
    if not m:
        return fallback_doc_key
    chapter = int(m.group(1))
    if 1 <= chapter <= 6:
        return f"ch{chapter}"
    return fallback_doc_key


def discover_doc_files() -> Dict[str, Path]:
    files = [p for p in DOC_ROOT.iterdir() if p.is_file() and p.suffix.lower() == ".docx" and not p.name.startswith("~$")]
    out: Dict[str, Path] = {}
    for key, patt in DOC_KEY_PATTERNS.items():
        for p in files:
            if patt in p.name:
                out[key] = p
                break
    missing = set(DOC_KEY_PATTERNS.keys()) - set(out.keys())
    if missing:
        raise RuntimeError(f"Missing doc files for: {sorted(missing)}")
    return out


def infer_kind(req_id: str, title: str, excerpt: str) -> Tuple[str, str]:
    req_type = "table" if req_id.startswith("表") else "figure"
    if req_type == "table":
        return req_type, "table"
    text = f"{title} {excerpt}"
    if any(k in text for k in FLOWCHART_KEYWORDS):
        return req_type, "flowchart"
    if any(k in text for k in HEATMAP_KEYWORDS):
        return req_type, "heatmap"
    if any(k in text for k in CURVE_KEYWORDS):
        return req_type, "curve"
    if any(k in text for k in IMAGE_PANEL_KEYWORDS):
        return req_type, "image_panel"
    if "示意图" in text:
        return req_type, "schematic"
    return req_type, "other"


def force_kind_for_priority(req_id: str, current_kind: str, title: str) -> str:
    # 某些图在段落中与“流程图”关键词共现，需按编号强制纠偏
    if req_id not in DATA_PRIORITY_IDS:
        return current_kind
    if req_id.startswith("表"):
        return "table"
    if "热力" in title or "风险图" in title:
        return "heatmap"
    if "曲线" in title or "对比" in title:
        return "curve"
    if "重建" in title or "分割" in title:
        return "image_panel"
    if "示意图" in title:
        return "schematic"
    return "other" if current_kind == "flowchart" else current_kind


def extract_visual(text: str) -> List[str]:
    elems: List[str] = []
    if "热力图" in text or "风险图" in text:
        elems += ["热力底图", "颜色映射", "风险分级"]
    if "曲线" in text:
        elems += ["二维坐标轴", "多曲线对比", "图例"]
    if "分割" in text:
        elems += ["原始影像", "语义掩膜", "类别标注"]
    if "重建" in text or "点云" in text:
        elems += ["三维点云", "空间分布", "颜色编码"]
    if "表" in text:
        elems += ["表头字段", "数据行", "对比维度"]
    out = []
    seen = set()
    for e in elems:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out


def extract_style(text: str) -> List[str]:
    out: List[str] = []
    if "红色" in text:
        out.append("包含红色高风险区域")
    if "蓝色" in text:
        out.append("包含蓝色安全区域")
    if "绿色" in text:
        out.append("包含绿色规划路径")
    if "改编自" in text:
        out.append("需保持与改编来源一致")
    if "创新图" in text:
        out.append("需体现新增创新元素")
    if "必须精确" in text:
        out.append("数值需精确")
    return out


def extract_size(text: str) -> List[str]:
    vals: List[str] = []
    for pat in (r"\d+(?:\.\d+)?\s*(?:m|km|cm|kPa|g|%)", r"\d+\s*[x×]\s*\d+"):
        vals.extend(re.findall(pat, text, flags=re.IGNORECASE))
    uniq = []
    seen = set()
    for v in vals:
        k = v.lower()
        if k not in seen:
            uniq.append(v)
            seen.add(k)
    return uniq


def expected_data(req_id: str, kind: str) -> List[str]:
    mapping = {
        "图M-2": ["DEM031202-200Img_X.tif"],
        "图2-2": ["DEM031202-200Img_X.tif", "ce4_easing_comparison_metrics.json", "ce4_panorama_quality_report.json"],
        "图3-3": ["DOM-031202-200_X1.tif"],
        "图3-4": ["DEM031202-200Img_X.tif"],
        "图5-3": ["DEM031202-200Img_X.tif"],
        "图5-4": ["DOM-031202-200_X1.tif", "DEM031202-200Img_X.tif"],
        "图6-3": ["DEM031202-200Img_X.tif"],
        "表6-1": ["ce4_easing_comparison_metrics.json"],
        "表3-5": ["ce4_easing_comparison_metrics.json", "derived_algorithm_scaling"],
    }
    if req_id in mapping:
        return mapping[req_id]
    if kind == "heatmap":
        return ["DEM031202-200Img_X.tif"]
    if kind == "table":
        return ["derived_table_data"]
    if kind in ("curve", "image_panel", "schematic"):
        return ["derived_model_data"]
    return []


def extract_requirements() -> List[Requirement]:
    docs = discover_doc_files()
    by_key: Dict[Tuple[str, str], Dict[str, object]] = {}

    for doc_key, doc_path in docs.items():
        doc = Document(str(doc_path))
        pars = [normalize_text(p.text) for p in doc.paragraphs if normalize_text(p.text)]
        for i, text in enumerate(pars):
            matches = list(REQ_PATTERN.finditer(text))
            if not matches:
                continue
            for j, m in enumerate(matches):
                req_id = f"{m.group(1)}{norm_dash(m.group(2).upper())}"
                seg_start = m.end()
                seg_end = matches[j + 1].start() if j + 1 < len(matches) else len(text)
                seg = normalize_text(text[seg_start:seg_end]).strip("：:，,。；;、* ")
                if not seg and i + 1 < len(pars):
                    seg = pars[i + 1][:220]
                context = " ".join(pars[max(0, i - 1) : min(len(pars), i + 2)])
                owner_key = canonical_doc_key(req_id, doc_key)
                ent = by_key.setdefault(
                    (owner_key, req_id),
                    {"doc_key": owner_key, "doc_name": docs.get(owner_key, doc_path).name, "req_id": req_id, "segments": [], "contexts": []},
                )
                if seg:
                    ent["segments"].append(seg)
                ent["contexts"].append(context)

    # Manual adds for total outline and chapter 2 concrete ids.
    for doc_key, req_id, title in [("outline", "图M-1", "跨章节接口契约总览"), ("outline", "图M-2", "任务预演风险热力图"), ("ch2", "图2-1", "UQ-DT理论框架图"), ("ch2", "图2-2", "接口数据契约矩阵图")]:
        k = (doc_key, req_id)
        if k not in by_key:
            by_key[k] = {"doc_key": doc_key, "doc_name": docs[doc_key].name, "req_id": req_id, "segments": [title], "contexts": [title]}

    reqs: List[Requirement] = []
    for (_doc_key, _rid), ent in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        req_id = ent["req_id"]
        segs = ent["segments"] or [TITLE_OVERRIDES.get(req_id, req_id)]
        ctxs = ent["contexts"] or segs
        title = TITLE_OVERRIDES.get(req_id, re.split(r"[。；;，,]", sorted(segs, key=len, reverse=True)[0])[0].strip())
        excerpt = sorted(ctxs, key=len, reverse=True)[0][:560]
        req_type, kind = infer_kind(req_id, title, excerpt)
        kind = force_kind_for_priority(req_id, kind, title)
        reqs.append(
            Requirement(
                doc_key=ent["doc_key"],
                doc_name=ent["doc_name"],
                req_id=req_id,
                req_type=req_type,
                title=title,
                excerpt=excerpt,
                chart_kind=kind,
                visual_elements=extract_visual(f"{title} {excerpt}"),
                style_requirements=extract_style(f"{title} {excerpt}"),
                size_specs=extract_size(excerpt),
                special_notes=[k for k in ("改编自附件论文", "创新图", "必须精确", "关键新增", "用于Agent生成", "对应附件论文") if k in excerpt],
                flowchart_deferred=(req_id in DEFERRED_FLOW_IDS or kind == "flowchart"),
                data_priority=(req_id in DATA_PRIORITY_IDS or (kind in {"heatmap", "curve", "table", "image_panel", "schematic"} and req_id not in DEFERRED_FLOW_IDS and kind != "flowchart")),
                expected_data=expected_data(req_id, kind),
                covered_by=(["图2-1", "图2-2"] if req_id == "图2-X" else []),
            )
        )
    return reqs


def write_outputs(reqs: List[Requirement]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in reqs], f, ensure_ascii=False, indent=2)

    lines = [
        "# 作图要求结构化清单",
        "",
        f"- 总条目数: {len(reqs)}",
        "- 说明: 从7个docx自动提取图/表要求，并标注视觉元素、风格、尺寸、特殊说明与优先级。",
        "",
        "| 文档 | 编号 | 类型 | 标题 | 类别 | 数据优先 | 流程图暂缓 | 预期数据 | 视觉元素 | 特殊说明 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reqs:
        exp = "<br>".join(r.expected_data) if r.expected_data else "-"
        vis = "<br>".join(r.visual_elements) if r.visual_elements else "-"
        notes = "<br>".join(r.special_notes) if r.special_notes else "-"
        lines.append(
            f"| {r.doc_key} | {r.req_id} | {r.req_type} | {r.title} | {r.chart_kind} | {r.data_priority} | {r.flowchart_deferred} | {exp} | {vis} | {notes} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    reqs = extract_requirements()
    reqs.sort(key=lambda x: (x.doc_key, x.req_id))
    write_outputs(reqs)
    print(f"requirements json: {OUT_JSON}")
    print(f"requirements md: {OUT_MD}")
    print(f"count: {len(reqs)}")


if __name__ == "__main__":
    main()
