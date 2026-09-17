# 环氧树脂 YAML + SMILES 审核工作流演示版

这是用于跨设备交流和课堂讨论的 Streamlit 云端镜像。它同步本地工作台的当前流程，但不读取或修改本地目录，也不包含本地数据库、API Key、备份或历史审核记录。

## 演示功能

- 内置一篇本地工作流中的已完成 YAML 示例。
- 上传多个 YAML，或选择文件夹递归导入。
- 导入时按 `paper_title` 去重。
- 当前 YAML 按“未完成”和“已完成”分开显示。
- 按材料类别、缩写和名称去重。
- 人工输入 SMILES，或从常用结构直接导入并进入下一条。
- 审核结果可撤回，输入的 SMILES 可快速加入常用结构。
- RDKit 合法性检查、Canonical SMILES 和二维结构图。
- 人工确认无适用单一 SMILES。
- 环氧单体、固化剂和填料常用结构的会话内添加与删除。
- 千问 AI 平台作为默认兼容接口，可切换模型、Base URL 或自定义接口，API Key 留空。
- 多体系并发模型分类，单项失败不影响其他结果。
- 无 API Key 时可生成明确标记的演示分类建议。
- 支持人工直接划分、模型建议和一键审核。
- conventional、degradable、mixed 拆分与分类 YAML 下载。
- 会话内 Canonical SMILES 去重结构列表。
- 文献管理、结果库、工作流状态和使用说明。

## 无持久化说明

应用只使用 `st.session_state`。上传内容、审核状态、RDKit 图片、常用结构、API 配置和输出 YAML 都只存在于当前 Streamlit 会话中，不写入数据库或服务器文件。

刷新会话、重启应用或云端实例休眠后，内容会消失。仓库内只有 `examples/completed_example.yaml` 这一份已完成示例，论文标题随数据保留。请勿在公开部署中上传保密论文或输入正式生产密钥。

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
