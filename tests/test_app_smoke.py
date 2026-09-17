from streamlit.testing.v1 import AppTest

from cloud_config import QIANWEN_BASE_URL, QIANWEN_MULTIMODAL_MODELS


def test_cloud_demo_opens_with_completed_example_and_creates_working_copy():
    app = AppTest.from_file("app.py", default_timeout=60).run(timeout=60)
    assert not app.exception
    assert sorted(doc["status"] for doc in app.session_state["docs"].values()) == [
        "COMPLETE"
    ]

    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert sorted(doc["status"] for doc in app.session_state["docs"].values()) == [
        "COMPLETE",
        "PENDING",
    ]
    assert len(app.dataframe) == 1


def test_qianwen_defaults_are_present_without_an_api_key():
    assert QIANWEN_BASE_URL == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert len(QIANWEN_MULTIMODAL_MODELS) >= 80
