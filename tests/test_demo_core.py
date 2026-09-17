from pathlib import Path

from demo_core import (
    SYSTEM_PROMPT,
    build_classified_outputs,
    classify_concurrently,
    load_yaml_text,
    parse_materials,
    update_material_smiles,
    validate_smiles,
)


ROOT = Path(__file__).resolve().parents[1]


def test_completed_example_parses_and_deduplicates_shared_materials():
    document = load_yaml_text((ROOT / "examples" / "completed_example.yaml").read_text(encoding="utf-8"))
    materials = parse_materials(document)
    assert len(materials) == 3
    dgeba = next(item for item in materials if item["abbreviation"] == "DGEBA")
    assert len(dgeba["paths"]) == 4
    update_material_smiles(document, dgeba, "CCO")
    assert all(
        system["epoxy_resin"][0]["SMILES"] == "CCO"
        for system in document["epoxy_resin_systems"]
    )


def test_rdkit_validation_and_mixed_output_are_in_memory():
    assert validate_smiles("CCO")["valid"]
    document = load_yaml_text((ROOT / "examples" / "completed_example.yaml").read_text(encoding="utf-8"))
    outputs = build_classified_outputs(
        document,
        {
            0: {"classification": "conventional_epoxy"},
            1: {
                "classification": "degradable_epoxy",
                "dynamic_bond_full_name": "二硫键",
                "dynamic_bond_resource": "填料：STP",
            },
            2: {"classification": "conventional_epoxy"},
            3: {"classification": "conventional_epoxy"},
        },
        "completed_example.yaml",
    )
    assert len(outputs) == 3
    assert all(isinstance(content, bytes) for content in outputs.values())


def test_concurrent_results_keep_success_when_one_request_fails():
    def worker(index):
        if index == 1:
            raise RuntimeError("failed")
        return ({"classification": "conventional_epoxy"}, "raw")

    results, errors = classify_concurrently([0, 1, 2], worker, max_workers=3)
    assert set(results) == {0, 2}
    assert errors == {1: "failed"}


def test_prompt_ignores_empty_dynamic_bond_placeholders_and_requires_chinese():
    assert "dynamic_bond_types: 无、none、null" in SYSTEM_PROMPT
    assert "自然语言必须使用中文" in SYSTEM_PROMPT
