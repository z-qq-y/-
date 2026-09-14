# 环氧树脂 YAML + SMILES 审核工作流演示版

这是用于跨设备交流和课堂讨论的 Streamlit 演示仓库。它从本地生产工具独立复制而来，不读取或修改生产版目录，也不包含真实论文、数据库、API Key 或历史审核记录。

## 演示功能

- 内置一篇明确标记为虚构数据的 conventional / degradable 混合示例。
- 上传 YAML 到浏览器当前会话。
- 当前 YAML 按“未完成”和“已完成”分开显示。
- 按材料类别、缩写和名称去重。
- 人工输入 SMILES，或从常用结构导入。
- RDKit 合法性检查、Canonical SMILES 和二维结构图。
- 人工确认无适用单一 SMILES。
- 环氧单体、固化剂和填料常用结构的会话内添加与删除。
- 可配置 Base URL、模型 ID 和 API Key。
- 多体系并发模型分类，单项失败不影响其他结果。
- 无 API Key 时可生成明确标记的演示分类建议。
- 逐体系人工确认、mixed 拆分与分类 YAML 下载。
- 会话内 Canonical SMILES 去重结构列表。

## 无持久化说明

应用只使用 `st.session_state`。上传内容、审核状态、RDKit 图片、常用结构、API 配置和输出 YAML 都只存在于当前 Streamlit 会话中，不写入数据库或服务器文件。

刷新会话、重启应用或云端实例休眠后，内容会消失。仓库内只有 `examples/demo.yaml` 这一份虚构示例。请勿在公开部署中上传保密论文或输入正式生产密钥。

## 本地运行

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## 发布到 GitHub

```powershell
git init
git add .
git commit -m "Add session-only epoxy review workflow demo"
git branch -M main
git remote add origin https://github.com/你的账号/epoxy-review-workflow-demo.git
git push -u origin main
```

本目录交付时已经初始化为本地 Git 仓库并创建首个提交。如果已有远程空仓库，只需要执行最后两行。

## 部署为在线演示

可在 Streamlit Community Cloud 中选择该 GitHub 仓库，入口文件填写 `app.py`。该演示不需要配置 secrets；只有实际测试大语言模型接口时才在页面中临时输入 API Key。

## 测试

```powershell
python -m pytest -q
```

## 与生产版的区别

演示版不包含 SQLite 工作流状态、永久结构库、文件备份、恢复和磁盘归档。正式科研数据处理仍应使用本地生产版，并按论文分组划分训练集与测试集。
