#!/usr/bin/env python3
"""
Audit generated figures and create per-image technical documentation.

Inputs:
- output/doc/2026-02-07_figures/docs/plot_requirements_structured.json
- output/doc/2026-02-07_figures/figure_generation_report_v2.json

Outputs:
- output/doc/2026-02-07_figures/docs/image_compliance_audit.json
- output/doc/2026-02-07_figures/docs/image_compliance_audit.md
- output/doc/2026-02-07_figures/docs/image_generation_technical_details.json
- output/doc/2026-02-07_figures/docs/image_generation_technical_details.md
- output/doc/2026-02-07_figures/docs/per_image/*.md
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "output" / "doc" / "2026-02-07_figures"
DOCS_ROOT = OUT_ROOT / "docs"
PER_IMAGE_DIR = DOCS_ROOT / "per_image"

REQ_JSON = DOCS_ROOT / "plot_requirements_structured.json"
GEN_JSON = OUT_ROOT / "figure_generation_report_v2.json"

AUDIT_JSON = DOCS_ROOT / "image_compliance_audit.json"
AUDIT_MD = DOCS_ROOT / "image_compliance_audit.md"
TECH_JSON = DOCS_ROOT / "image_generation_technical_details.json"
TECH_MD = DOCS_ROOT / "image_generation_technical_details.md"


def image_size(path: Path):
    if not path.exists():
        return None, None
    with Image.open(path) as im:
        return int(im.size[0]), int(im.size[1])


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def classify(requirement: dict, gen: dict) -> Dict[str, object]:
    req_id = requirement["req_id"]
    flow_deferred = requirement.get("flowchart_deferred", False)
    expected_data = requirement.get("expected_data", [])
    status = gen.get("status", "missing")
    ascii_file = gen.get("ascii_file")
    cn_file = gen.get("cn_file")

    width = gen.get("width")
    height = gen.get("height")
    if width is None or height is None:
        if ascii_file:
            w, h = image_size(Path(ascii_file))
            width, height = w, h

    # 审核规则
    if req_id == "图2-X":
        cstatus = "deferred_abstract"
        notes = "抽象编号由图2-1/图2-2覆盖，不单独审核图片"
    elif flow_deferred and status.startswith("deferred_flowchart"):
        cstatus = "deferred_flowchart"
        notes = "流程图按优先级暂缓"
    elif status == "generated":
        if not ascii_file or not Path(ascii_file).exists():
            cstatus = "fail"
            notes = "生成记录存在但图片文件缺失"
        elif width is None or height is None:
            cstatus = "fail"
            notes = "图片尺寸读取失败"
        elif width < 800 or height < 500:
            cstatus = "fail"
            notes = f"分辨率不足 ({width}x{height})"
        else:
            # 数据优先项需要至少可追踪到数据源或明确派生
            data_sources = gen.get("data_sources", [])
            if requirement.get("data_priority", False) and not data_sources and not expected_data:
                cstatus = "fail"
                notes = "数据优先项缺少数据来源与派生说明"
            else:
                cstatus = "pass"
                notes = "满足当前作图要求并通过基础审核"
    elif status in ("deferred_flowchart_missing", "generate_failed", "missing"):
        cstatus = "fail"
        notes = f"状态异常: {status}"
    else:
        # 其他状态统一按失败处理，避免漏审
        cstatus = "fail"
        notes = f"未识别状态: {status}"

    return {
        "req_id": req_id,
        "doc_key": requirement.get("doc_key"),
        "doc_name": requirement.get("doc_name"),
        "title": requirement.get("title"),
        "chart_kind": requirement.get("chart_kind"),
        "data_priority": requirement.get("data_priority"),
        "expected_data": expected_data,
        "generation_status": status,
        "compliance_status": cstatus,
        "compliance_notes": notes,
        "ascii_file": ascii_file,
        "cn_file": cn_file,
        "width": width,
        "height": height,
        "data_driven": gen.get("data_driven", False),
        "data_sources": gen.get("data_sources", []),
        "technical": gen.get("technical", {}),
    }


def write_per_image_docs(records: List[dict]) -> None:
    PER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for r in records:
        fn = r["req_id"].replace("-", "_") + ".md"
        path = PER_IMAGE_DIR / fn
        tech = r.get("technical", {}) or {}
        lines = [
            f"# {r['req_id']} {r.get('title','')}",
            "",
            f"- 文档来源: `{r.get('doc_name','')}`",
            f"- 图片类别: `{r.get('chart_kind','')}`",
            f"- 生成状态: `{r.get('generation_status','')}`",
            f"- 合规状态: `{r.get('compliance_status','')}`",
            f"- 合规说明: {r.get('compliance_notes','')}",
            f"- 数据优先项: `{r.get('data_priority', False)}`",
            f"- 数据驱动: `{r.get('data_driven', False)}`",
            f"- ASCII图片: `{r.get('ascii_file')}`",
            f"- 中文图片: `{r.get('cn_file')}`",
            f"- 分辨率: `{r.get('width')}x{r.get('height')}`",
            "",
            "## 原始数据",
        ]
        if r.get("data_sources"):
            lines += [f"- {s}" for s in r["data_sources"]]
        else:
            lines.append("- 无直接数据源（基于描述参数推导）")
        lines += ["", "## 数据处理过程"]
        if tech.get("processing"):
            lines += [f"- {x}" for x in tech["processing"]]
        else:
            lines.append("- 未提供处理步骤")
        lines += ["", "## 生成逻辑", f"- {tech.get('key_impl', '未提供')}"]
        lines += ["", "## 参数设置", "```json", json.dumps(tech.get("parameters", {}), ensure_ascii=False, indent=2), "```", ""]
        lines += ["## 关键技术实现细节", f"- {tech.get('note', 'NumPy/Pillow/Matplotlib组合绘制')}"]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_markdown_audit(records: List[dict]) -> str:
    lines = [
        "# 图片合规审核报告",
        "",
        f"- 审核条目: {len(records)}",
        "- `pass`: 已按要求生成并通过审核",
        "- `deferred_flowchart`: 流程图按优先级暂缓",
        "- `deferred_abstract`: 抽象编号要求由具体图覆盖",
        "- `fail`: 缺失或不符合要求",
        "",
        "| 编号 | 文档 | 状态 | 生成状态 | 数据驱动 | 分辨率 | 说明 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        res = f"{r.get('width','-')}x{r.get('height','-')}" if r.get("width") else "-"
        lines.append(
            f"| {r['req_id']} | {r['doc_key']} | {r['compliance_status']} | {r['generation_status']} | {r['data_driven']} | {res} | {r['compliance_notes']} |"
        )
    lines.append("")
    return "\n".join(lines)


def to_markdown_tech(records: List[dict]) -> str:
    lines = ["# 逐图技术文档汇总", ""]
    for r in records:
        lines.append(f"## {r['req_id']} {r.get('title','')}")
        lines.append(f"- 合规状态: `{r['compliance_status']}`")
        lines.append(f"- 数据源: {', '.join(r.get('data_sources', [])) if r.get('data_sources') else '无（参数推导）'}")
        lines.append(f"- 生成逻辑: {r.get('technical', {}).get('key_impl', '未提供')}")
        lines.append(f"- 参数: `{json.dumps(r.get('technical', {}).get('parameters', {}), ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not REQ_JSON.exists():
        raise FileNotFoundError(f"Missing requirements: {REQ_JSON}")
    if not GEN_JSON.exists():
        raise FileNotFoundError(f"Missing generation report: {GEN_JSON}")

    reqs = load_json(REQ_JSON)
    gens = load_json(GEN_JSON)
    by_gen = {g["req_id"]: g for g in gens}

    records: List[dict] = []
    for req in reqs:
        gen = by_gen.get(req["req_id"], {"status": "missing"})
        records.append(classify(req, gen))

    # 排序：先失败，再通过，再暂缓，便于检查
    order = {"fail": 0, "pass": 1, "deferred_flowchart": 2, "deferred_abstract": 3}
    records.sort(key=lambda x: (order.get(x["compliance_status"], 9), x["doc_key"], x["req_id"]))

    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    with AUDIT_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    AUDIT_MD.write_text(to_markdown_audit(records), encoding="utf-8")

    with TECH_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    TECH_MD.write_text(to_markdown_tech(records), encoding="utf-8")
    write_per_image_docs(records)

    stats = Counter(r["compliance_status"] for r in records)
    print(f"audit json: {AUDIT_JSON}")
    print(f"audit md: {AUDIT_MD}")
    print(f"tech json: {TECH_JSON}")
    print(f"tech md: {TECH_MD}")
    print(f"per-image docs dir: {PER_IMAGE_DIR}")
    print(f"stats: {dict(stats)}")


if __name__ == "__main__":
    main()
