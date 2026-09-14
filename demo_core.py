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
只返回 JSON。classification 只能是 conventional_epoxy 或 degradable_epoxy。
若为 degradable_epoxy，填写 dynamic_bond_full_name 和 dynamic_bond_resource。
若证据不足，requires_human_review 必须为 true。不要仅凭 SMILES 中的官能团判断。
JSON 字段必须包含 classification、confidence、requires_human_review、reasoning_summary、dynamic_bond_full_name、dynamic_bond_resource。
"""


def call_openai_compatible(
    base_url: str,
    model: str,
    api_key: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
