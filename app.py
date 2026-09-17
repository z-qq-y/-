from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from cloud_config import (
    QIANWEN_BASE_URL,
    QIANWEN_DEFAULT_MODEL,
    QIANWEN_MULTIMODAL_MODELS,
)

from demo_core import (
    SYSTEM_PROMPT,
    build_classified_outputs,
    call_openai_compatible,
    classify_concurrently,
    dump_yaml_bytes,
    load_yaml_text,
    material_keys_for_system,
    parse_materials,
    update_material_smiles,
    validate_smiles,
)


BASE_DIR = Path(__file__).resolve().parent
EXAMPLE_PATH = BASE_DIR / "examples" / "completed_example.yaml"
PROCESS_PAGES = ["材料审核", "体系分类", "完成归档"]
COMMON_CATEGORIES = ["环氧单体", "固化剂", "填料"]
CATEGORY_MAP = {
    "epoxy_resin": "环氧单体",
    "curing_agents": "固化剂",
    "additives": "填料",
}
MODEL_LABELS = {
    model_id: f"{category} | {model_id} | {description}"
    for category, model_id, description in QIANWEN_MULTIMODAL_MODELS
}
MODEL_IDS = tuple(MODEL_LABELS)
CUSTOM_MODEL_OPTION = "__custom_model__"

st.set_page_config(page_title="环氧树脂数据工作流演示", layout="wide")
st.markdown(
    """
    <style>
    :root {--ink:#1d2b38;--muted:#607286;--line:#d6e0e8;--accent:#245d89;--panel:#fbfcfe;}
    [data-testid="stAppViewContainer"] {background:#f5f7fa;}
    [data-testid="stSidebar"] {background:#eef3f7;border-right:1px solid var(--line);}
    .block-container {max-width:1380px;padding-top:2rem;padding-bottom:4rem;}
    .hero {padding:1.2rem 1.35rem;margin-bottom:1.3rem;border:1px solid var(--line);
      border-left:4px solid var(--accent);border-radius:12px;background:var(--panel);}
    .hero h1 {margin:0 0 .3rem;color:var(--ink);font-size:clamp(1.45rem,2.4vw,2.05rem);}
    .hero p {margin:0;color:var(--muted);}
    .section {margin:1rem 0 .35rem;color:#526578;font-size:.72rem;font-weight:700;
      letter-spacing:.08em;text-transform:uppercase;}
    [data-testid="stMetric"], [data-testid="stExpander"] {border:1px solid var(--line);
      border-radius:12px;background:var(--panel);}
    [data-testid="stMetric"] {padding:.8rem 1rem;}
    .stButton>button,.stDownloadButton>button {border-radius:8px;font-weight:600;}
    [data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:12px;overflow:hidden;}
    @media (max-width: 768px) {
      .block-container {padding:1rem .85rem 3rem;}
      .hero {padding:1rem;}
      [data-testid="stHorizontalBlock"] {gap:.55rem;}
      [data-testid="stMetric"] {min-width:0;}
    }
    @media (prefers-reduced-motion: reduce) {
      *,*::before,*::after {transition:none !important;scroll-behavior:auto !important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_title(value: str) -> str:
    return " ".join(value.split()).casefold()


def material_smiles(document: dict, material: dict):
    node = document
    for segment in material["paths"][0]:
        node = node[segment]
    return node.get("SMILES")


def add_document(name: str, text: str, allow_duplicate: bool = False) -> dict:
    document = load_yaml_text(text)
    title = str(document.get("paper_title") or "").strip()
    if not title:
        raise ValueError("YAML 缺少 paper_title。")
    if not allow_duplicate:
        duplicate = next(
            (
                doc
                for doc in st.session_state.docs.values()
                if normalize_title(doc["title"]) == normalize_title(title)
            ),
            None,
        )
        if duplicate:
            return {
                "status": "标题重复",
                "name": Path(name).name,
                "title": title,
                "message": f"已存在于 {duplicate['name']}。",
            }
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    doc_id = f"{Path(name).stem}-{digest}"
    record = {
        "id": doc_id,
        "name": Path(name).name,
        "title": title,
        "document": document,
        "materials": parse_materials(document),
        "reviews": {},
        "proposals": {},
        "classifications": {},
        "structures": {},
        "outputs": {},
        "status": "PENDING",
    }
    st.session_state.docs[doc_id] = record

    for material in record["materials"]:
        smiles = material_smiles(document, material)
        if smiles in (None, "", "null", "None"):
            continue
        if str(smiles).strip().casefold() in {"none", "无", "not applicable", "n/a"}:
            record["reviews"][material["material_key"]] = {
                "status": "HUMAN_CONFIRMED_NO_SMILES",
                "no_smiles": True,
            }
            continue
        result = validate_smiles(str(smiles))
        if result["valid"]:
            record["reviews"][material["material_key"]] = {
                "status": "HUMAN_CONFIRMED",
                "smiles": str(smiles),
                "canonical": result["canonical_smiles"],
                "image": result["image_png"],
            }
            record["structures"].setdefault(
                result["canonical_smiles"],
                {"smiles": str(smiles), "sources": []},
            )["sources"].append(material["full_name"])

    for index, system in enumerate(document.get("epoxy_resin_systems", [])):
        classification = system.get("classification")
        if classification not in {"conventional_epoxy", "degradable_epoxy"}:
            continue
        dynamic = system.get("dynamic_bond_types")
        dynamic = dynamic if isinstance(dynamic, dict) else {}
        record["classifications"][index] = {
            "classification": classification,
            "dynamic_bond_full_name": (
                dynamic.get("full_name") if classification == "degradable_epoxy" else None
            ),
            "dynamic_bond_resource": (
                dynamic.get("resource") if classification == "degradable_epoxy" else None
            ),
            "reasoning_summary": "已完成 YAML 中的人工分类结果",
            "classification_method": "imported",
        }
    refresh_status(record)
    if (
        len(record["reviews"]) == len(record["materials"])
        and len(record["classifications"]) == len(record["document"]["epoxy_resin_systems"])
    ):
        record["outputs"] = build_classified_outputs(
            record["document"], record["classifications"], record["name"]
        )
        record["status"] = "COMPLETE"
    return {"status": "成功", "name": record["name"], "title": title, "id": doc_id}


def initialize_state() -> None:
    if "docs" not in st.session_state:
        st.session_state.docs = {}
    if "common_structures" not in st.session_state:
        seeds = {
            "环氧单体": [
                ("双酚 A 环氧树脂", "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1")
            ],
            "固化剂": [("4,4'-二氨基二苯甲烷", "Nc1ccc(Cc2ccc(N)cc2)cc1")],
            "填料": [("二氧化硅", "O=[Si]=O")],
        }
        library = {category: [] for category in COMMON_CATEGORIES}
        for category, rows in seeds.items():
            for index, (name, smiles) in enumerate(rows, start=1):
                result = validate_smiles(smiles)
                if result["valid"]:
                    library[category].append(
                        {
                            "id": f"seed-{category}-{index}",
                            "name": name,
                            "smiles": smiles,
                            "canonical": result["canonical_smiles"],
                            "image": result["image_png"],
                        }
                    )
        st.session_state.common_structures = library
    if not st.session_state.docs:
        add_document(EXAMPLE_PATH.name, EXAMPLE_PATH.read_text(encoding="utf-8"))
    st.session_state.setdefault("page", "已完成 YAML")
    st.session_state.setdefault("process", "材料审核")
    st.session_state.setdefault("last_process", "材料审核")
    st.session_state.setdefault("base_url", QIANWEN_BASE_URL)
    st.session_state.setdefault("provider", "千问 AI 平台（默认）")
    st.session_state.setdefault("model_choice", QIANWEN_DEFAULT_MODEL)
    st.session_state.setdefault("custom_model", "")
    st.session_state.setdefault("api_key", "")
    st.session_state.setdefault("system_prompt", SYSTEM_PROMPT)
    st.session_state.setdefault("import_report", [])
    st.session_state.setdefault("retained_structures", {})


def navigate(page: str) -> None:
    st.session_state.page = page
    if page in PROCESS_PAGES:
        st.session_state.process = page
        st.session_state.last_process = page


def apply_provider() -> None:
    if st.session_state.provider == "千问 AI 平台（默认）":
        st.session_state.base_url = QIANWEN_BASE_URL
        st.session_state.model_choice = QIANWEN_DEFAULT_MODEL
        st.session_state.custom_model = ""


def selected_model() -> str:
    if st.session_state.model_choice == CUSTOM_MODEL_OPTION:
        return str(st.session_state.custom_model).strip()
    return str(st.session_state.model_choice)


def create_review_copy(source: dict) -> None:
    document = copy.deepcopy(source["document"])
    document["paper_title"] = source["title"] + "（审核演示副本）"
    for system in document.get("epoxy_resin_systems", []):
        for category in ("epoxy_resin", "curing_agents", "additives"):
            materials = system.get(category) or []
            if isinstance(materials, dict):
                materials = [materials]
            for material in materials:
                if isinstance(material, dict):
                    material["SMILES"] = None
        system.pop("classification", None)
        system["dynamic_bond_types"] = None
    result = add_document(
        f"{Path(source['name']).stem}_review_demo.yaml",
        dump_yaml_bytes(document).decode("utf-8"),
    )
    if result["status"] == "成功":
        st.session_state.page = "材料审核"


def refresh_status(doc: dict) -> None:
    if doc["status"] == "COMPLETE":
        return
    material_done = len(doc["reviews"]) == len(doc["materials"])
    class_done = len(doc["classifications"]) == len(doc["document"]["epoxy_resin_systems"])
    if material_done and class_done:
        doc["status"] = "LLM_CLASSIFIED"
    elif material_done:
        doc["status"] = "HUMAN_CONFIRMED"
    elif doc["reviews"]:
        doc["status"] = "HUMAN_CONFIRMED_PARTIAL"
    else:
        doc["status"] = "PENDING"


def progress(doc: dict) -> None:
    material_done = len(doc["reviews"])
    material_total = len(doc["materials"])
    system_done = len(doc["classifications"])
    system_total = len(doc["document"]["epoxy_resin_systems"])
    first, second, third = st.columns(3)
    first.metric("材料审核", f"{material_done} / {material_total}")
    second.metric("体系分类", f"{system_done} / {system_total}")
    third.metric("归档状态", "已完成" if doc["status"] == "COMPLETE" else "未完成")


def store_review(doc: dict, material: dict, smiles: str, result: dict) -> None:
    update_material_smiles(doc["document"], material, smiles)
    doc["reviews"][material["material_key"]] = {
        "status": "HUMAN_CONFIRMED",
        "smiles": smiles,
        "canonical": result["canonical_smiles"],
        "image": result["image_png"],
    }
    sources = doc["structures"].setdefault(
        result["canonical_smiles"], {"smiles": smiles, "sources": []}
    )["sources"]
    if material["full_name"] not in sources:
        sources.append(material["full_name"])
    refresh_status(doc)


def material_page(doc: dict) -> None:
    st.header("材料审核")
    progress(doc)
    materials = doc["materials"]
    table = []
    for item in materials:
        review = doc["reviews"].get(item["material_key"])
        table.append(
            {
                "材料": item["full_name"],
                "缩写": item["abbreviation"],
                "类别": item["category"],
                "出现体系": "、".join(item["system_names"]),
                "状态": review["status"] if review else "PENDING",
            }
        )
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    index_key = f"material-index-{doc['id']}"
    st.session_state.setdefault(index_key, 0)
    index = min(st.session_state[index_key], len(materials) - 1)
    material = materials[index]
    review = doc["reviews"].get(material["material_key"])
    st.subheader(f"材料 {index + 1} / {len(materials)}：{material['full_name']}")
    st.caption("出现体系：" + "、".join(material["system_names"]))

    if review:
        if review.get("no_smiles"):
            st.info("已确认该材料无适用的单一 SMILES。")
        else:
            st.success("已完成 RDKit 验证和人工确认。")
            st.code(review["smiles"], language=None)
            st.image(review["image"], width=560)
        with st.expander("撤回本条审核"):
            withdraw = st.checkbox(
                "我确认撤回此材料的审核结果",
                key=f"withdraw-check-{doc['id']}-{material['material_key']}",
            )
            if st.button(
                "撤回并重新审核",
                disabled=not withdraw,
                key=f"withdraw-{doc['id']}-{material['material_key']}",
            ):
                canonical = review.get("canonical")
                update_material_smiles(doc["document"], material, None)
                doc["reviews"].pop(material["material_key"], None)
                if canonical in doc["structures"]:
                    sources = doc["structures"][canonical]["sources"]
                    doc["structures"][canonical]["sources"] = [
                        source for source in sources if source != material["full_name"]
                    ]
                    if not doc["structures"][canonical]["sources"]:
                        doc["structures"].pop(canonical, None)
                for path in material["paths"]:
                    system_index = path[1]
                    doc["classifications"].pop(system_index, None)
                    doc["document"]["epoxy_resin_systems"][system_index].pop("classification", None)
                    doc["document"]["epoxy_resin_systems"][system_index]["dynamic_bond_types"] = None
                doc["outputs"] = {}
                doc["status"] = "PENDING"
                refresh_status(doc)
                st.rerun()
    else:
        mode = st.radio(
            "处理方式",
            ["输入 SMILES", "该材料无适用 SMILES"],
            horizontal=True,
            key=f"mode-{doc['id']}-{material['material_key']}",
        )
        if mode == "输入 SMILES":
            smiles_key = f"smiles-{doc['id']}-{material['material_key']}"
            st.session_state.setdefault(smiles_key, "")
            category = CATEGORY_MAP.get(material["category"], "环氧单体")
            common = st.session_state.common_structures[category]
            with st.expander("从常用结构导入"):
                if common:
                    labels = {f"{item['name']} | {item['smiles']}": item for item in common}
                    label = st.selectbox("选择结构", list(labels), key=f"common-{doc['id']}-{index}")
                    selected = labels[label]
                    st.image(selected["image"], width=340)
                    if st.button(
                        "使用并直接进入下一条",
                        type="primary",
                        width="stretch",
                        key=f"use-{doc['id']}-{index}",
                    ):
                        result = validate_smiles(selected["smiles"])
                        store_review(doc, material, selected["smiles"], result)
                        st.session_state[index_key] = min(index + 1, len(materials) - 1)
                        st.rerun()
                else:
                    st.info("此分类尚无常用结构。")
            smiles = st.text_area("SMILES", key=smiles_key, height=105)
            validation_key = f"validation-{doc['id']}-{material['material_key']}"
            if st.button("RDKit 验证", type="primary", key=f"validate-{doc['id']}-{index}"):
                st.session_state[validation_key] = validate_smiles(smiles)
            result = st.session_state.get(validation_key)
            if result and result.get("valid") and result.get("input_smiles") == smiles:
                st.success("RDKit 验证通过。请继续核对论文结构。")
                st.code(result["canonical_smiles"], language=None)
                st.image(result["image_png"], width=620)
                confirm = st.checkbox(
                    "我确认与论文结构一致",
                    key=f"confirm-{doc['id']}-{material['material_key']}",
                )
                add_common = st.checkbox(
                    "同时快速加入常用结构",
                    key=f"add-common-{doc['id']}-{material['material_key']}",
                )
                common_name = ""
                if add_common:
                    common_name = st.text_input(
                        "常用结构名称",
                        value=material["abbreviation"] or material["full_name"],
                        key=f"common-name-{doc['id']}-{material['material_key']}",
                    )
                if st.button(
                    "保存并下一条",
                    disabled=not confirm or (add_common and not common_name.strip()),
                    width="stretch",
                ):
                    store_review(doc, material, smiles, result)
                    if add_common:
                        exists = any(
                            item["canonical"] == result["canonical_smiles"]
                            for item in st.session_state.common_structures[category]
                        )
                        if not exists:
                            st.session_state.common_structures[category].append(
                                {
                                    "id": hashlib.sha1(
                                        f"{category}-{common_name}-{smiles}".encode()
                                    ).hexdigest()[:10],
                                    "name": common_name.strip(),
                                    "smiles": smiles,
                                    "canonical": result["canonical_smiles"],
                                    "image": result["image_png"],
                                }
                            )
                    st.session_state[index_key] = min(index + 1, len(materials) - 1)
                    st.rerun()
            elif result and not result.get("valid"):
                st.error(result.get("error"))
        else:
            confirm_none = st.checkbox(
                "我确认该材料没有适用的单一 SMILES",
                key=f"none-{doc['id']}-{material['material_key']}",
            )
            if st.button("保存无 SMILES 并下一条", disabled=not confirm_none, width="stretch"):
                update_material_smiles(doc["document"], material, "none")
                doc["reviews"][material["material_key"]] = {
                    "status": "HUMAN_CONFIRMED_NO_SMILES",
                    "no_smiles": True,
                }
                refresh_status(doc)
                st.session_state[index_key] = min(index + 1, len(materials) - 1)
                st.rerun()

    complete = len(doc["reviews"]) == len(materials)
    if complete:
        st.success("材料审核已完成。")
        previous, skip, next_column, advance = st.columns(4)
    else:
        previous, skip, next_column = st.columns(3)
    if previous.button("上一条", disabled=index == 0, width="stretch"):
        st.session_state[index_key] = index - 1
        st.rerun()
    if skip.button("暂时跳过", width="stretch"):
        st.session_state[index_key] = (index + 1) % len(materials)
        st.rerun()
    if next_column.button("下一条", disabled=index == len(materials) - 1, width="stretch"):
        st.session_state[index_key] = index + 1
        st.rerun()
    if complete:
        advance.button("进入体系分类", type="primary", width="stretch", on_click=navigate, args=("体系分类",))


def demo_proposal(index: int) -> dict:
    return {
        "classification": "conventional_epoxy",
        "confidence": 0.91,
        "requires_human_review": False,
        "reasoning_summary": "演示判断：当前体系未报告可逆动态键、解聚或受控降解机制。",
        "dynamic_bond_full_name": None,
        "dynamic_bond_resource": None,
    }


def classification_page(doc: dict) -> None:
    st.header("体系分类")
    progress(doc)
    systems = doc["document"]["epoxy_resin_systems"]
    all_materials_done = len(doc["reviews"]) == len(doc["materials"])
    if not all_materials_done:
        st.warning("请先完成全部材料审核。模型不是必选项，材料完成后可人工直接划分。")
        st.button("返回材料审核", width="stretch", on_click=navigate, args=("材料审核",))
        return
    mode = st.radio(
        "分类方式",
        ["人工直接划分", "模型辅助分类"],
        horizontal=True,
        key=f"classification-mode-{doc['id']}",
        help="两种方式都需要用户最终确认。模型不是必选项。",
    )
    pending = [index for index in range(len(systems)) if index not in doc["classifications"]]
    labels = {
        index: f"{index + 1}. {systems[index].get('abbreviation') or systems[index].get('full_name')}"
        for index in range(len(systems))
    }
    if pending and mode == "模型辅助分类":
        st.info(
            f"当前接口：{st.session_state.base_url}　模型：{selected_model() or '未选择'}。"
            "API Key 只存在于当前会话。"
        )
        batch_key = f"batch-{doc['id']}"
        if batch_key not in st.session_state:
            st.session_state[batch_key] = pending
        else:
            st.session_state[batch_key] = [
                index for index in st.session_state[batch_key] if index in pending
            ]
        selected = st.multiselect(
            "选择同时分类的体系",
            pending,
            format_func=lambda index: labels[index],
            key=batch_key,
        )
        workers = st.number_input("最大并发数", 1, 8, min(4, len(pending)), key=f"workers-{doc['id']}")
        demo_column, api_column = st.columns(2)
        if demo_column.button("生成演示分类建议", width="stretch", disabled=not selected):
            for index in selected:
                doc["proposals"][index] = demo_proposal(index)
            st.rerun()
        if api_column.button(
            f"并发调用模型（{len(selected)} 个体系）",
            type="primary",
            width="stretch",
            disabled=not selected or not st.session_state.api_key or not selected_model(),
        ):
            def worker(index: int):
                payload = {
                    "paper_metadata": {
                        key: value
                        for key, value in doc["document"].items()
                        if key != "epoxy_resin_systems"
                    },
                    "epoxy_resin_system": systems[index],
                }
                return call_openai_compatible(
                    st.session_state.base_url,
                    selected_model(),
                    st.session_state.api_key,
                    payload,
                    st.session_state.system_prompt,
                )

            with st.spinner("正在并发分类……"):
                results, errors = classify_concurrently(selected, worker, int(workers))
            for index, (proposal, _raw) in results.items():
                doc["proposals"][index] = proposal
            if results:
                st.success(f"已收到 {len(results)} 个分类建议。")
            for index, error in errors.items():
                st.error(f"{labels[index]}：{error}")

    if pending:
        st.subheader("一键审核")
        st.caption("可直接人工填写，也可修改模型建议。点击一次按钮会写入表中全部待分类体系。")
        rows = []
        for index in pending:
            proposal = doc["proposals"].get(index)
            proposal = proposal or demo_proposal(index)
            rows.append(
                {
                    "体系索引": index,
                    "体系": labels[index],
                    "最终分类": proposal["classification"],
                    "动态键中文全名": proposal.get("dynamic_bond_full_name") or "",
                    "动态键材料来源": proposal.get("dynamic_bond_resource") or "",
                    "中文判断依据": (
                        proposal.get("reasoning_summary")
                        if index in doc["proposals"]
                        else "人工直接分类"
                    ),
                    "复核提示": (
                        "重点复核" if proposal.get("requires_human_review") else "常规复核"
                    ),
                }
            )
        edited = st.data_editor(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            disabled=["体系索引", "体系", "复核提示"],
            column_config={
                "最终分类": st.column_config.SelectboxColumn(
                    "最终分类",
                    options=["conventional_epoxy", "degradable_epoxy"],
                    required=True,
                )
            },
            key=f"classification-editor-{doc['id']}",
        )
        if st.button(
            f"一键审核并写入 {len(pending)} 个体系",
            type="primary",
            width="stretch",
            key=f"batch-confirm-{doc['id']}",
        ):
            invalid = []
            prepared = []
            for _, row in edited.iterrows():
                dynamic_name = "" if pd.isna(row["动态键中文全名"]) else str(row["动态键中文全名"]).strip()
                resource = "" if pd.isna(row["动态键材料来源"]) else str(row["动态键材料来源"]).strip()
                if row["最终分类"] == "degradable_epoxy" and (not dynamic_name or not resource):
                    invalid.append(str(row["体系"]))
                prepared.append((int(row["体系索引"]), row, dynamic_name, resource))
            if invalid:
                st.error("以下可降解体系缺少动态键或材料来源：" + "、".join(invalid))
            else:
                for index, row, dynamic_name, resource in prepared:
                    classification = str(row["最终分类"])
                    if classification == "conventional_epoxy":
                        dynamic_name = ""
                        resource = ""
                    doc["classifications"][index] = {
                        "classification": classification,
                        "dynamic_bond_full_name": dynamic_name or None,
                        "dynamic_bond_resource": resource or None,
                        "reasoning_summary": str(row["中文判断依据"]),
                        "classification_method": (
                            "model_assisted" if index in doc["proposals"] else "manual"
                        ),
                    }
                    systems[index]["classification"] = classification
                    systems[index]["dynamic_bond_types"] = (
                        {"full_name": dynamic_name, "resource": resource}
                        if classification == "degradable_epoxy"
                        else None
                    )
                    doc["proposals"].pop(index, None)
                refresh_status(doc)
                st.rerun()

    for index, system in enumerate(systems):
        row = doc["classifications"].get(index)
        with st.expander(labels[index] + " | " + (row["classification"] if row else "待分类")):
            st.write(system.get("full_name") or "")
            if row:
                st.success("已人工确认：" + row["classification"])
                if row.get("dynamic_bond_full_name"):
                    st.write("动态键：" + row["dynamic_bond_full_name"])
                    st.write("材料来源：" + str(row.get("dynamic_bond_resource") or ""))
                if st.button("撤回本体系分类", key=f"withdraw-class-{doc['id']}-{index}"):
                    doc["classifications"].pop(index, None)
                    system.pop("classification", None)
                    system["dynamic_bond_types"] = None
                    doc["outputs"] = {}
                    doc["status"] = "HUMAN_CONFIRMED"
                    refresh_status(doc)
                    st.rerun()

    if len(doc["classifications"]) == len(systems):
        st.success("体系分类已经全部人工确认。")
        st.button("进入完成检查与归档", type="primary", width="stretch", on_click=navigate, args=("完成归档",))


def completion_page(doc: dict) -> None:
    st.header("完成检查与分类归档")
    progress(doc)
    if len(doc["reviews"]) != len(doc["materials"]):
        st.warning("仍有材料未审核。")
        return
    if len(doc["classifications"]) != len(doc["document"]["epoxy_resin_systems"]):
        st.warning("仍有体系未确认分类。")
        return
    if doc["status"] != "COMPLETE" and st.button(
        "完成归档", type="primary", width="stretch"
    ):
        doc["outputs"] = build_classified_outputs(
            doc["document"], doc["classifications"], doc["name"]
        )
        doc["status"] = "COMPLETE"
        next_doc = next(
            (
                candidate
                for candidate in st.session_state.docs.values()
                if candidate["id"] != doc["id"]
                and candidate["status"] == "PENDING"
                and not candidate["reviews"]
                and not candidate["classifications"]
            ),
            None,
        )
        if next_doc:
            st.session_state.selected_unfinished_id = next_doc["id"]
            st.session_state.page = "材料审核"
        else:
            st.session_state.selected_completed_id = doc["id"]
            st.session_state.page = "已完成 YAML"
        st.rerun()
    if doc["outputs"]:
        st.success("演示归档已完成。文件只存在于当前会话，可直接下载。")
        for output_name, content in doc["outputs"].items():
            st.download_button(
                f"下载 {output_name}",
                data=content,
                file_name=Path(output_name).name,
                mime="application/yaml",
                key=f"output-{doc['id']}-{output_name}",
            )


def common_page() -> None:
    st.header("常用结构")
    tabs = st.tabs(COMMON_CATEGORIES)
    for tab, category in zip(tabs, COMMON_CATEGORIES):
        with tab:
            with st.form(f"add-common-{category}", clear_on_submit=True):
                name = st.text_input("名称")
                smiles = st.text_area("SMILES")
                submitted = st.form_submit_button("RDKit 验证并添加")
            if submitted:
                result = validate_smiles(smiles)
                if not name.strip():
                    st.error("名称不能为空。")
                elif not result["valid"]:
                    st.error(result["error"])
                else:
                    st.session_state.common_structures[category].append(
                        {
                            "id": hashlib.sha1(f"{category}-{name}-{smiles}".encode()).hexdigest()[:10],
                            "name": name.strip(),
                            "smiles": smiles,
                            "canonical": result["canonical_smiles"],
                            "image": result["image_png"],
                        }
                    )
                    st.rerun()
            for item in list(st.session_state.common_structures[category]):
                with st.expander(item["name"]):
                    left, right = st.columns([1, 1.4])
                    left.image(item["image"], width=340)
                    right.code(item["smiles"], language=None)
                    if right.button("删除", key=f"delete-common-{item['id']}"):
                        st.session_state.common_structures[category].remove(item)
                        st.rerun()


def documents_page() -> None:
    st.header("文献管理")
    if not st.session_state.docs:
        st.info("当前会话没有文献。可从侧栏导入 YAML 或恢复内置示例。")
        return
    rows = []
    for doc in st.session_state.docs.values():
        rows.append(
            {
                "论文": doc["title"],
                "YAML": doc["name"],
                "状态": doc["status"],
                "材料": f"{len(doc['reviews'])}/{len(doc['materials'])}",
                "体系": f"{len(doc['classifications'])}/{len(doc['document']['epoxy_resin_systems'])}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    choices = {f"{doc['title']} 〔{doc['name']}〕": doc_id for doc_id, doc in st.session_state.docs.items()}
    selected = st.selectbox("选择文献", list(choices))
    doc_id = choices[selected]
    st.download_button(
        "下载当前会话 YAML",
        data=dump_yaml_bytes(st.session_state.docs[doc_id]["document"]),
        file_name=st.session_state.docs[doc_id]["name"],
        mime="application/yaml",
    )
    delete_mode = st.radio(
        "删除范围",
        ["仅删除文献记录，保留结构库信息", "删除文献记录和数据库信息"],
        key=f"delete-mode-{doc_id}",
    )
    confirm = st.checkbox("我确认从当前会话删除此文献")
    if st.button("删除会话文献", disabled=not confirm):
        if delete_mode.startswith("仅删除"):
            doc = st.session_state.docs[doc_id]
            for canonical, row in doc["structures"].items():
                target = st.session_state.retained_structures.setdefault(
                    canonical, {"smiles": row["smiles"], "sources": []}
                )
                target["sources"].extend(
                    f"{doc['title']}：{source}" for source in row["sources"]
                )
        del st.session_state.docs[doc_id]
        st.rerun()


def structures_page() -> None:
    st.header("结果库")
    yaml_tab, structure_tab = st.tabs(["分类 YAML", "化学结构库"])
    with yaml_tab:
        output_rows = [
            {"论文": doc["title"], "分类文件": output_name, "体系数": len(doc["document"]["epoxy_resin_systems"])}
            for doc in st.session_state.docs.values()
            if doc["status"] == "COMPLETE"
            for output_name in doc["outputs"]
        ]
        if output_rows:
            st.dataframe(pd.DataFrame(output_rows), hide_index=True, width="stretch")
        else:
            st.info("完成归档后会在这里显示分类 YAML。")
    with structure_tab:
        _structures_table()


def _structures_table() -> None:
    merged: dict[str, dict] = copy.deepcopy(st.session_state.retained_structures)
    for doc in st.session_state.docs.values():
        for canonical, row in doc["structures"].items():
            target = merged.setdefault(
                canonical,
                {"smiles": row["smiles"], "sources": []},
            )
            target["sources"].extend(
                f"{doc['title']}: {source}" for source in row["sources"]
            )
    if not merged:
        st.info("完成材料审核后，这里会按 Canonical SMILES 展示会话内去重结构。")
        return
    rows = [
        {"Canonical SMILES": canonical, "原始 SMILES": row["smiles"], "材料来源": "、".join(row["sources"])}
        for canonical, row in merged.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def workflow_page() -> None:
    st.header("工作流状态")
    rows = [
        {
            "论文": doc["title"],
            "YAML": doc["name"],
            "状态": doc["status"],
            "材料审核": f"{len(doc['reviews'])}/{len(doc['materials'])}",
            "体系分类": f"{len(doc['classifications'])}/{len(doc['document']['epoxy_resin_systems'])}",
        }
        for doc in st.session_state.docs.values()
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("当前会话没有文献记录。")


def help_page() -> None:
    st.header("使用说明与工作流")
    st.markdown(
        """
工作台按照“导入 YAML、人工提供 SMILES、RDKit 验证、人工确认、体系分类、分类归档”运行。它不会自动生成 SMILES，也不会替代用户对照论文原始结构。

1. 从左侧导入一个或多个 YAML，也可以选择文件夹递归导入。
2. 系统按 `paper_title` 去重，未完成和已完成 YAML 分开显示。
3. 材料审核可输入 SMILES、使用常用结构，或确认没有适用的单一 SMILES。
4. 体系分类可人工直接划分，也可并发调用模型并一键审核。
5. 归档结果进入 conventional、degradable，mixed 论文同时保留完整审核版本。

`导入 YAML → 材料审核 → RDKit 校验 → 人工确认 → 体系分类 → 分类归档`

### 云端演示边界

内置示例是本地工作流中的一份已完成 YAML，保留论文标题、摘要、SMILES 和体系分类。

所有用户输入、图片、API 配置和处理结果只保存在 Streamlit 当前会话中。云端版不连接本地 SQLite，不上传本地历史或备份。刷新会话、重启应用或云端实例休眠后，数据会消失。API Key 默认留空。
        """
    )
    st.subheader("工作流示意图")
    st.markdown(
        """
| 阶段 | 输入与动作 | 结果 |
|---|---|---|
| 导入 | 文件或文件夹，按论文标题去重 | 未完成 YAML |
| 材料审核 | SMILES、常用结构或 none | RDKit 结构确认 |
| 体系分类 | 人工直接划分或模型辅助 | conventional / degradable |
| 完成归档 | 拆分 mixed 并保留论文标题 | 分类 YAML + 化学结构库 |
        """
    )


initialize_state()
st.sidebar.markdown("### 环氧树脂数据工作台")
st.sidebar.caption("云端交流镜像。无数据库写入，会话结束即清空")

st.sidebar.markdown('<div class="section">工作文件</div>', unsafe_allow_html=True)
with st.sidebar.expander("导入 YAML"):
    import_mode = st.radio(
        "导入方式",
        ["选择 YAML 文件", "选择文件夹"],
        key="import_mode",
        label_visibility="collapsed",
    )
    if import_mode == "选择文件夹":
        uploads = st.file_uploader(
            "选择包含 YAML 的文件夹",
            type=["yaml", "yml"],
            accept_multiple_files="directory",
            key="folder_upload",
            help="会递归读取所选文件夹及其子文件夹中的 YAML。",
        )
        import_label = "导入文件夹中的 YAML"
    else:
        uploads = st.file_uploader(
            "选择一个或多个 YAML",
            type=["yaml", "yml"],
            accept_multiple_files=True,
            key="file_upload",
        )
        import_label = "导入所选 YAML"
    if st.button(import_label, disabled=not uploads, width="stretch"):
        report = []
        for upload in uploads or []:
            try:
                report.append(add_document(upload.name, upload.getvalue().decode("utf-8")))
            except Exception as exc:
                report.append({"status": "失败", "name": upload.name, "message": str(exc)})
        st.session_state.import_report = report
        st.rerun()
    for row in st.session_state.import_report:
        st.write(f"**{row['status']}**　{row['name']}")
        if row.get("message"):
            st.caption(row["message"])
    if st.button("恢复已完成示例", width="stretch"):
        try:
            result = add_document(EXAMPLE_PATH.name, EXAMPLE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            result = {"status": "失败", "name": EXAMPLE_PATH.name, "message": str(exc)}
        st.session_state.import_report = [result]
        st.rerun()

with st.sidebar.expander("模型 API 配置"):
    st.selectbox(
        "接口预设",
        ["千问 AI 平台（默认）", "自定义 OpenAI 兼容接口"],
        key="provider",
        on_change=apply_provider,
    )
    st.text_input("Base URL", key="base_url")
    st.selectbox(
        "选择多模态模型",
        [*MODEL_IDS, CUSTOM_MODEL_OPTION],
        key="model_choice",
        format_func=lambda model_id: (
            "手动输入其他模型 ID"
            if model_id == CUSTOM_MODEL_OPTION
            else MODEL_LABELS[model_id]
        ),
    )
    if st.session_state.model_choice == CUSTOM_MODEL_OPTION:
        st.text_input("自定义模型 ID", key="custom_model")
    st.caption(f"已列出 {len(MODEL_IDS)} 个多模态 Chat 模型。")
    st.text_input("API Key（仅当前会话）", key="api_key", type="password")
    st.text_area("体系分类提示词（仅当前会话）", key="system_prompt", height=300)
    if st.button("恢复默认提示词", width="stretch"):
        st.session_state.system_prompt = SYSTEM_PROMPT
        st.rerun()

unfinished = [doc for doc in st.session_state.docs.values() if doc["status"] != "COMPLETE"]
completed = [doc for doc in st.session_state.docs.values() if doc["status"] == "COMPLETE"]
st.sidebar.markdown('<div class="section">当前 YAML</div>', unsafe_allow_html=True)
st.sidebar.caption(f"未完成 {len(unfinished)}　已完成 {len(completed)}")
selected_unfinished = None
selected_completed = None
with st.sidebar.expander(f"未完成 YAML（{len(unfinished)}）", expanded=bool(unfinished)):
    if unfinished:
        unfinished_choices = {
            f"{doc['title']} 〔{doc['name']}〕": doc for doc in unfinished
        }
        selected_unfinished = unfinished_choices[
            st.selectbox("选择未完成 YAML", list(unfinished_choices), label_visibility="collapsed")
        ]
    else:
        st.caption("没有未完成文件")
with st.sidebar.expander(f"已完成 YAML（{len(completed)}）"):
    if completed:
        completed_choices = {
            f"{doc['title']} 〔{doc['name']}〕": doc for doc in completed
        }
        selected_completed = completed_choices[
            st.selectbox("选择已完成 YAML", list(completed_choices), label_visibility="collapsed")
        ]
        st.button("查看已完成 YAML", width="stretch", on_click=navigate, args=("已完成 YAML",))
    else:
        st.caption("没有已完成文件")

st.sidebar.markdown('<div class="section">处理程序</div>', unsafe_allow_html=True)
with st.sidebar.expander("打开处理程序"):
    for label in PROCESS_PAGES:
        st.button(
            label,
            type="primary" if st.session_state.page == label else "secondary",
            width="stretch",
            on_click=navigate,
            args=(label,),
            key=f"process-nav-{label}",
        )

st.sidebar.markdown('<div class="section">常用结构</div>', unsafe_allow_html=True)
st.sidebar.button("打开常用结构", width="stretch", on_click=navigate, args=("常用结构",))
st.sidebar.markdown('<div class="section">数据库与状态</div>', unsafe_allow_html=True)
database_a, database_b = st.sidebar.columns(2)
database_a.button("结果库", width="stretch", on_click=navigate, args=("结果库",))
database_b.button("工作流", width="stretch", on_click=navigate, args=("工作流",))
st.sidebar.markdown('<div class="section">文献管理</div>', unsafe_allow_html=True)
st.sidebar.button("打开文献管理", width="stretch", on_click=navigate, args=("文献管理",))
st.sidebar.markdown('<div class="section">使用说明</div>', unsafe_allow_html=True)
st.sidebar.button("查看使用说明", width="stretch", on_click=navigate, args=("使用说明",))

page = st.session_state.page

st.markdown(
    """
    <div class="hero"><h1>环氧树脂数据工作台</h1>
    <p>文献 YAML 审核、结构确认与体系分类。SMILES 由用户提供，论文结构一致性由用户确认。</p></div>
    """,
    unsafe_allow_html=True,
)

if page == "文献管理":
    documents_page()
elif page == "常用结构":
    common_page()
elif page == "结果库":
    structures_page()
elif page == "工作流":
    workflow_page()
elif page == "使用说明":
    help_page()
elif page == "已完成 YAML":
    current_doc = selected_completed
    if current_doc is None:
        st.info("当前会话没有已完成 YAML。")
    else:
        st.header("已完成 YAML")
        progress(current_doc)
        st.success("示例来自本地工作流的已完成记录，论文标题、摘要、SMILES 和分类结果均保留。")
        st.write(f"**论文：** {current_doc['title']}")
        if current_doc["document"].get("abstract"):
            with st.expander("查看原文摘要"):
                st.write(current_doc["document"]["abstract"])
        for output_name, content in current_doc["outputs"].items():
            st.download_button(
                f"下载 {output_name}",
                data=content,
                file_name=Path(output_name).name,
                mime="application/yaml",
            )
        st.divider()
        st.write("要演示完整审核流程，可从这份已完成 YAML 创建会话内工作副本。")
        if st.button("创建可操作的审核演示副本", type="primary", width="stretch"):
            create_review_copy(current_doc)
            st.rerun()
elif not selected_unfinished:
    st.info("当前没有未完成 YAML。可打开已完成示例并创建审核演示副本。")
elif page == "材料审核":
    material_page(selected_unfinished)
elif page == "体系分类":
    classification_page(selected_unfinished)
elif page == "完成归档":
    completion_page(selected_unfinished)
else:
    st.header("已完成 YAML")
    current_doc = selected_completed
    if current_doc:
        progress(current_doc)
        st.info("已完成文献与未完成文献分开显示。")
    else:
        st.info("当前会话没有已完成 YAML。")
