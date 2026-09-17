from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO, StringIO
from typing import Any, Callable

from openai import OpenAI
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from ruamel.yaml import YAML


MATERIAL_CATEGORIES = ("epoxy_resin", "curing_agents", "additives")


def load_yaml_text(text: str) -> dict[str, Any]:
    yaml = YAML(typ="safe")
    document = yaml.load(text)
    if not isinstance(document, dict):
        raise ValueError("YAML 顶层必须是映射。")
    systems = document.get("epoxy_resin_systems")
    if not isinstance(systems, list):
        raise ValueError("缺少 epoxy_resin_systems 列表。")
    return document


def dump_yaml_bytes(document: dict[str, Any]) -> bytes:
    yaml = YAML()
    yaml.allow_unicode = True
    yaml.default_flow_style = False
    stream = StringIO()
    yaml.dump(document, stream)
    return stream.getvalue().encode("utf-8")


def parse_materials(document: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for system_index, system in enumerate(document.get("epoxy_resin_systems", [])):
        if not isinstance(system, dict):
            continue
        system_name = str(
            system.get("abbreviation")
            or system.get("full_name")
            or f"体系 #{system_index + 1}"
        )
        for category in MATERIAL_CATEGORIES:
            materials = system.get(category) or []
            if isinstance(materials, dict):
                materials = [materials]
            if not isinstance(materials, list):
                continue
            for material_index, material in enumerate(materials):
                if not isinstance(material, dict):
                    continue
                name = str(material.get("full_name") or "未命名材料")
                abbreviation = str(material.get("abbreviation") or "")
                identity = f"{category}|{abbreviation.casefold()}|{name.casefold()}"
                key = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
                row = grouped.setdefault(
                    key,
                    {
                        "material_key": key,
                        "category": category,
                        "role": str(material.get("role") or ""),
                        "full_name": name,
                        "abbreviation": abbreviation,
                        "paths": [],
                        "system_names": [],
                    },
                )
                row["paths"].append(
                    ("epoxy_resin_systems", system_index, category, material_index)
                )
                if system_name not in row["system_names"]:
                    row["system_names"].append(system_name)
    return list(grouped.values())


def material_keys_for_system(materials: list[dict[str, Any]], index: int) -> list[str]:
    return [
        item["material_key"]
        for item in materials
        if any(path[1] == index for path in item["paths"])
    ]


def update_material_smiles(
    document: dict[str, Any],
    material: dict[str, Any],
    smiles: str | None,
) -> None:
    for path in material["paths"]:
        node: Any = document
        for segment in path:
            node = node[segment]
        node["SMILES"] = smiles


def validate_smiles(smiles: str) -> dict[str, Any]:
    if not smiles or not smiles.strip():
        return {"valid": False, "error": "SMILES 不能为空。"}
    try:
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
        if molecule is None:
            return {"valid": False, "error": "RDKit 无法解析该 SMILES。"}
        Chem.SanitizeMol(molecule)
        canonical = Chem.MolToSmiles(molecule, canonical=True)
        AllChem.Compute2DCoords(molecule)
        image = Draw.MolToImage(molecule, size=(720, 420))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return {
            "valid": True,
            "input_smiles": smiles,
            "canonical_smiles": canonical,
            "image_png": buffer.getvalue(),
        }
    except Exception as exc:
        return {"valid": False, "error": f"RDKit 校验失败：{exc}"}


SYSTEM_PROMPT = """你负责对单个环氧树脂体系进行严格二分类。
判断对象与边界：
1. 只判断输入中 epoxy_resin_system 所代表的当前体系，不要用整篇论文的总体主题替代当前体系判断。
2. 一篇讨论可降解材料的论文也可能包含普通环氧树脂对照组，必须逐体系判断。
3. 只允许返回 conventional_epoxy 或 degradable_epoxy。conventional_epoxy 表示当前证据没有证明该体系可降解，不表示该材料在所有条件下绝对不可降解。

证据规则：
1. 优先依据摘要、当前体系组成和明确的实验描述中关于降解、解聚、回收或动态交换机制的证据。
2. 若摘要与当前体系的组成或明确实验描述冲突，以当前体系的组成和明确实验描述为准。
3. 不能仅因某个 SMILES 含有酯键等官能团，或名称中出现“生物基”“绿色”“可回收”等词语，就判为可降解。
4. 输入 YAML 中原本存在的 dynamic_bond_types: 无、none、null、空字符串、not applicable 等内容均视为待补全占位值。它们既不是存在动态键的证据，也不是不存在动态键的证据，不得据此判断分类。
5. 若证据不足，仍返回最可能的二分类，同时将 requires_human_review 设为 true，并在 reasoning_summary 中用中文说明缺少什么证据。

动态键信息：
1. 若为 degradable_epoxy，dynamic_bond_full_name 填当前体系中有明确依据的动态键中文全名；dynamic_bond_resource 填实际产生该动态键的当前体系材料类别和材料名称，例如“固化剂：XXX（ABC）”。resource 不是论文出处。
2. 动态键必须能对应到当前体系中的具体材料；无法对应时 requires_human_review 必须为 true。
3. 若为 conventional_epoxy，dynamic_bond_full_name 和 dynamic_bond_resource 必须为 null。

输出语言：
classification 字段保留规定的英文枚举值。reasoning_summary、dynamic_bond_full_name 和 dynamic_bond_resource 中的自然语言必须使用中文，不要返回英文说明。

必须只返回一个 JSON 对象，不要使用 Markdown 代码块或添加 JSON 之外的文字。JSON 必须包含 classification、confidence、requires_human_review、reasoning_summary、dynamic_bond_full_name、dynamic_bond_resource 六个字段。
"""


def call_openai_compatible(
    base_url: str,
    model: str,
    api_key: str,
    payload: dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[dict[str, Any], str]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    proposal = json.loads(raw)
    if proposal.get("classification") not in {
        "conventional_epoxy",
        "degradable_epoxy",
    }:
        raise ValueError("模型返回了不支持的分类。")
    return proposal, raw


def classify_concurrently(
    indices: list[int],
    worker: Callable[[int], tuple[dict[str, Any], str]],
    max_workers: int = 4,
) -> tuple[dict[int, tuple[dict[str, Any], str]], dict[int, str]]:
    unique_indices = list(dict.fromkeys(indices))
    if not unique_indices:
        return {}, {}
    results: dict[int, tuple[dict[str, Any], str]] = {}
    errors: dict[int, str] = {}
    with ThreadPoolExecutor(
        max_workers=max(1, min(max_workers, len(unique_indices)))
    ) as executor:
        futures = {executor.submit(worker, index): index for index in unique_indices}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                errors[index] = str(exc)
    return results, errors


def build_classified_outputs(
    document: dict[str, Any],
    classifications: dict[int, dict[str, Any]],
    original_name: str,
) -> dict[str, bytes]:
    systems = document.get("epoxy_resin_systems", [])
    if len(classifications) != len(systems):
        raise ValueError("仍有体系未确认分类。")
    reviewed = copy.deepcopy(document)
    for index, system in enumerate(reviewed["epoxy_resin_systems"]):
        proposal = classifications[index]
        label = proposal["classification"]
        system["classification"] = label
        if label == "degradable_epoxy":
            system["dynamic_bond_types"] = {
                "full_name": proposal.get("dynamic_bond_full_name"),
                "resource": proposal.get("dynamic_bond_resource"),
            }
        else:
            system["dynamic_bond_types"] = None

    labels = {item["classification"] for item in classifications.values()}
    stem = original_name.rsplit(".", 1)[0]
    if len(labels) == 1:
        folder = "degradable" if "degradable_epoxy" in labels else "conventional"
        return {f"{folder}/{original_name}": dump_yaml_bytes(reviewed)}

    conventional = copy.deepcopy(reviewed)
    conventional["epoxy_resin_systems"] = [
        system
        for index, system in enumerate(reviewed["epoxy_resin_systems"])
        if classifications[index]["classification"] == "conventional_epoxy"
    ]
    degradable = copy.deepcopy(reviewed)
    degradable["epoxy_resin_systems"] = [
        system
        for index, system in enumerate(reviewed["epoxy_resin_systems"])
        if classifications[index]["classification"] == "degradable_epoxy"
    ]
    return {
        f"conventional/{stem}_conventional.yaml": dump_yaml_bytes(conventional),
        f"degradable/{stem}_degradable.yaml": dump_yaml_bytes(degradable),
        f"mixed_source/{stem}_reviewed.yaml": dump_yaml_bytes(reviewed),
    }
