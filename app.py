from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from demo_core import (
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
EXAMPLE_PATH = BASE_DIR / "examples" / "demo.yaml"
PROCESS_PAGES = ["材料审核", "体系分类", "完成归档"]
COMMON_CATEGORIES = ["环氧单体", "固化剂", "填料"]
CATEGORY_MAP = {
    "epoxy_resin": "环氧单体",
    "curing_agents": "固化剂",
    "additives": "填料",
}

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
    </style>
    """,
    unsafe_allow_html=True,
)


def add_document(name: str, text: str) -> str:
    document = load_yaml_text(text)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    doc_id = f"{Path(name).stem}-{digest}"
    st.session_state.docs[doc_id] = {
        "id": doc_id,
        "name": Path(name).name,
        "title": str(document.get("paper_title") or Path(name).stem),
        "document": document,
        "materials": parse_materials(document),
        "reviews": {},
        "proposals": {},
        "classifications": {},
        "structures": {},
        "outputs": {},
        "status": "PENDING",
    }
    return doc_id


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
    st.session_state.setdefault("page", "材料审核")
    st.session_state.setdefault("process", "材料审核")
    st.session_state.setdefault("last_process", "材料审核")
    st.session_state.setdefault("base_url", "https://api.moonshot.cn/v1")
    st.session_state.setdefault("model", "kimi-k3")
    st.session_state.setdefault("api_key", "")


def navigate(page: str) -> None:
    st.session_state.page = page
    if page in PROCESS_PAGES:
        st.session_state.process = page
        st.session_state.last_process = page


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
                    if st.button("使用此结构", key=f"use-{doc['id']}-{index}"):
                        st.session_state[smiles_key] = selected["smiles"]
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
                if st.button("保存并下一条", disabled=not confirm, width="stretch"):
                    update_material_smiles(doc["document"], material, smiles)
                    doc["reviews"][material["material_key"]] = {
                        "status": "HUMAN_CONFIRMED",
                        "smiles": smiles,
                        "canonical": result["canonical_smiles"],
                        "image": result["image_png"],
                    }
                    doc["structures"].setdefault(
                        result["canonical_smiles"],
                        {"smiles": smiles, "sources": []},
                    )["sources"].append(material["full_name"])
                    refresh_status(doc)
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
                update_material_smiles(doc["document"], material, None)
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
    if index == 1:
        return {
            "classification": "degradable_epoxy",
            "confidence": 0.88,
            "requires_human_review": False,
            "reasoning_summary": "演示结果：该体系含二硫键固化剂并描述了动态交换。",
            "dynamic_bond_full_name": "disulfide bond",
            "dynamic_bond_resource": "curing_agents: 4-aminophenyl disulfide (4-AFD)",
        }
    return {
        "classification": "conventional_epoxy",
        "confidence": 0.91,
        "requires_human_review": False,
        "reasoning_summary": "演示结果：未报告可逆键或受控降解机制。",
        "dynamic_bond_full_name": None,
        "dynamic_bond_resource": None,
    }


def classification_page(doc: dict) -> None:
    st.header("体系分类")
    progress(doc)
    systems = doc["document"]["epoxy_resin_systems"]
    all_materials_done = len(doc["reviews"]) == len(doc["materials"])
    if not all_materials_done:
        st.warning("请先完成全部材料审核。也可使用下面的演示建议预览界面。")
    pending = [index for index in range(len(systems)) if index not in doc["classifications"]]
    labels = {
        index: f"{index + 1}. {systems[index].get('abbreviation') or systems[index].get('full_name')}"
        for index in pending
    }
    if pending:
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
            disabled=not selected or not st.session_state.api_key,
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
                    st.session_state.model,
                    st.session_state.api_key,
                    payload,
                )

            with st.spinner("正在并发分类……"):
                results, errors = classify_concurrently(selected, worker, int(workers))
            for index, (proposal, _raw) in results.items():
                doc["proposals"][index] = proposal
            if results:
                st.success(f"已收到 {len(results)} 个分类建议。")
            for index, error in errors.items():
                st.error(f"{labels[index]}：{error}")

    for index, system in enumerate(systems):
        name = system.get("abbreviation") or system.get("full_name") or f"体系 #{index + 1}"
        with st.expander(f"{index + 1}. {name}", expanded=index not in doc["classifications"]):
            if index in doc["classifications"]:
                st.success("已人工确认：" + doc["classifications"][index]["classification"])
                continue
            proposal = doc["proposals"].get(index)
            if not proposal:
                st.info("等待模型建议或演示建议。")
                continue
            st.write(proposal.get("reasoning_summary", ""))
            with st.form(f"classification-{doc['id']}-{index}"):
                options = ["conventional_epoxy", "degradable_epoxy"]
                classification = st.selectbox(
                    "最终分类",
                    options,
                    index=options.index(proposal["classification"]),
                )
                dynamic_name = st.text_input(
                    "动态键",
                    value=proposal.get("dynamic_bond_full_name") or "",
                )
                resource = st.text_input(
                    "动态键材料来源",
                    value=proposal.get("dynamic_bond_resource") or "",
                )
                confirmed = st.checkbox("我已人工复核")
                submitted = st.form_submit_button("确认分类")
            if submitted:
                if not confirmed:
                    st.error("请先完成人工复核确认。")
                elif classification == "degradable_epoxy" and (not dynamic_name or not resource):
                    st.error("可降解体系必须填写动态键和材料来源。")
                else:
                    doc["classifications"][index] = {
                        "classification": classification,
                        "dynamic_bond_full_name": dynamic_name or None,
                        "dynamic_bond_resource": resource or None,
                    }
                    refresh_status(doc)
                    st.rerun()

    if len(doc["classifications"]) == len(systems):
        st.success("体系分类已经全部人工确认。")
        st.button("进入完成归档", type="primary", width="stretch", on_click=navigate, args=("完成归档",))


def completion_page(doc: dict) -> None:
    st.header("完成归档")
    progress(doc)
    if len(doc["reviews"]) != len(doc["materials"]):
        st.warning("仍有材料未审核。")
        return
    if len(doc["classifications"]) != len(doc["document"]["epoxy_resin_systems"]):
        st.warning("仍有体系未确认分类。")
        return
    if doc["status"] != "COMPLETE" and st.button("在会话中生成分类 YAML", type="primary"):
        doc["outputs"] = build_classified_outputs(
            doc["document"], doc["classifications"], doc["name"]
        )
        doc["status"] = "COMPLETE"
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
    confirm = st.checkbox("我确认从当前会话删除此文献")
    if st.button("删除会话文献", disabled=not confirm):
        del st.session_state.docs[doc_id]
        st.rerun()


def structures_page() -> None:
    st.header("化学结构库")
    merged: dict[str, dict] = {}
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


def help_page() -> None:
    st.header("演示说明")
    st.markdown(
        """
本仓库是交流演示版。它保留 YAML 导入、材料去重、SMILES 人工审核、RDKit 校验与绘图、常用结构、并发模型分类、人工确认、mixed 拆分和 YAML 下载。

所有用户输入、图片、API 配置和处理结果只保存在 Streamlit 当前会话中。刷新会话、重启应用或云端实例休眠后，数据会消失。请勿把真实保密论文或正式 API Key 用于公开演示。
        """
    )


initialize_state()
st.sidebar.markdown("### 环氧树脂工作流演示")
st.sidebar.caption("无数据库、无文件写入、会话结束即清空")
st.sidebar.markdown('<div class="section">处理程序</div>', unsafe_allow_html=True)
process = st.sidebar.radio("处理程序", PROCESS_PAGES, key="process", label_visibility="collapsed")
if process != st.session_state.last_process:
    st.session_state.page = process
    st.session_state.last_process = process
page = st.session_state.page

unfinished = [doc for doc in st.session_state.docs.values() if doc["status"] != "COMPLETE"]
completed = [doc for doc in st.session_state.docs.values() if doc["status"] == "COMPLETE"]
current_doc = None
if page in PROCESS_PAGES and st.session_state.docs:
    st.sidebar.markdown('<div class="section">当前 YAML</div>', unsafe_allow_html=True)
    groups = (["未完成 YAML"] if unfinished else []) + (["已完成 YAML"] if completed else [])
    scope = st.sidebar.radio("YAML 状态", groups, horizontal=True, label_visibility="collapsed")
    pool = unfinished if scope == "未完成 YAML" else completed
    choices = {f"{doc['title']} 〔{doc['name']}〕": doc for doc in pool}
    current_doc = choices[st.sidebar.selectbox(scope, list(choices))]

for label in ["文献管理", "常用结构", "化学结构库", "使用说明"]:
    st.sidebar.button(label, width="stretch", on_click=navigate, args=(label,))

st.sidebar.divider()
uploads = st.sidebar.file_uploader("导入 YAML 到当前会话", type=["yaml", "yml"], accept_multiple_files=True)
if st.sidebar.button("导入", disabled=not uploads, width="stretch"):
    for upload in uploads or []:
        add_document(upload.name, upload.getvalue().decode("utf-8"))
    st.rerun()
if st.sidebar.button("恢复内置示例", width="stretch"):
    add_document(EXAMPLE_PATH.name, EXAMPLE_PATH.read_text(encoding="utf-8"))
    st.rerun()

with st.sidebar.expander("模型 API 配置"):
    st.text_input("Base URL", key="base_url")
    st.text_input("模型 ID", key="model")
    st.text_input("API Key", key="api_key", type="password")
    st.caption("密钥默认留空，只存在于当前会话。")

st.markdown(
    """
    <div class="hero"><h1>环氧树脂数据工作流演示</h1>
    <p>用于跨设备交流的无持久化版本。所有更改只存在于当前浏览器会话。</p></div>
    """,
    unsafe_allow_html=True,
)

if page == "文献管理":
    documents_page()
elif page == "常用结构":
    common_page()
elif page == "化学结构库":
    structures_page()
elif page == "使用说明":
    help_page()
elif not current_doc:
    st.info("请导入 YAML 或恢复内置示例。")
elif current_doc["status"] == "COMPLETE":
    st.header("已完成 YAML")
    progress(current_doc)
    st.info("已完成文献与未完成文献分开显示，此处为只读演示视图。")
    for output_name, content in current_doc["outputs"].items():
        st.download_button(
            f"下载 {output_name}", data=content, file_name=Path(output_name).name, mime="application/yaml"
        )
elif page == "材料审核":
    material_page(current_doc)
elif page == "体系分类":
    classification_page(current_doc)
else:
    completion_page(current_doc)
