# LangGraph MOF Chatbot 🤖

基于 LangGraph + LangChain + DashScope 构建的科研型 RAG Chatbot，能够针对金属有机框架（MOF）文献知识进行检索增强问答（RAG）。

---

## 🧠 系统概述

LangGraph MOF Chatbot 能够读取、分块、嵌入并索引科研文献，支持用户以自然语言查询本地语料，返回包含来源编号的结构化答案。

### 架构流程

```
CLI (app/cli.py)
   ↳ GraphRunner → 输出格式化
Graph (app/graph.py)
   parse_query → retrieve_docs → maybe_call_tools_node → generate
RAG 数据层 (app/rag/)
   ingest.py, retriever.py, memory.py
```

---

## ⚙️ 技术栈

- **LangGraph**：状态图控制对话流
- **LangChain + Chroma**：语料切分、向量化与存储
- **DashScope API**：通义千问模型接口
- **Typer + Rich**：命令行交互
- **pypdf + TextSplitter**：文档解析与分块

---

## 🚀 使用方法

### 1️⃣ 安装依赖
```bash
pip install -r requirements.txt
```

### 2️⃣ 构建向量数据库
```bash
python -m app.rag.ingest --input-dir ./data/samples --persist-dir ./.chroma_mof
```

### 3️⃣ 启动 Chatbot
```bash
python -m app.cli --persist-dir ./.chroma_mof
```

---

## 💡 示例对话
```text
You > UiO-66 药物负载的典型范围和影响因素？
Bot >
[LOCAL] 10–30 wt% ...
[INFERRED] 孔径匹配推断 ...
[PRIOR] 药理学因素（不在本地语料）
```

---

## 📁 项目结构
```
mofbot/
│
├── app/
│   ├── cli.py
│   ├── graph.py
│   ├── config.py
│   ├── rag/
│   │   ├── ingest.py
│   │   ├── retriever.py
│   │   └── memory.py
│   └── tools/
│
├── data/samples/
│   ├── UiO-66_drug_delivery.md
│   ├── ZIF8_CO2_capture.md
│   └── ...
│
├── .chroma_mof/
└── requirements.txt
```

---

## 🔗 链接

- 📘 项目报告：[LangGraph_MOF_Chatbot_Report_CN.pdf](./LangGraph_MOF_Chatbot_Report_CN.pdf)
- 🌐 DashScope 官网：[https://dashscope.aliyun.com](https://dashscope.aliyun.com)

---

## 🧾 License
MIT License © 2025 Junzhe Zha
