# ScholarSeeker Web 演示

一个能真正搜索论文的前后端 Web 应用。

## 结构

```
code/web_server.py   后端服务（FastAPI，复用 src/ 的检索 pipeline）
demo/index.html      前端页面（输入查询 → 调后端 → 展示结果）
```

## 启动步骤

### 1. 启动后端

```bash
cd D:\desktop\文献检索\code
python web_server.py
```

后端启动在 `http://127.0.0.1:8000`，提供接口：
- `POST /search`　请求体 `{"query": "你的研究问题"}`
- `GET /docs`　Swagger 接口文档

> 后端依赖 `config/.env` 里的 `DEEPSEEK_API_KEY`（查询理解、约束自省、精排用大模型）和 Semantic Scholar（免费，无需 key）。

### 2. 打开前端

用浏览器打开 `demo/index.html`（直接双击即可），输入研究问题，点击「搜索」。

## 搜索流程

一次搜索会依次执行：

```
查询理解（Planner）→ 迭代检索（Retriever + Critic 自省补检索）→ 排序（Selector）→ 结构化展示
```

## 注意事项

1. **需要大模型 API key**：`config/.env` 里配好 `DEEPSEEK_API_KEY`（或 DashScope/Qwen）。
2. **Semantic Scholar 限流**：免费层 100 req/5min，一次搜索会调用约 20~50 次 API，频繁搜索会撞限流。
3. **耗时**：一次完整搜索约 10~40 秒（取决于迭代轮数和 API 响应），前端会显示「正在检索」状态。
4. **Embedding 模型**：向量检索用本地 MiniLM（`config/.env` 里的 `EMBEDDING_MODEL_PATH_LIGHT`），未配置会降级为 TF-IDF。

## 常见问题

- **前端提示「无法连接后端服务」**：后端没启动，先跑 `python web_server.py`。
- **后端报错「DEEPSEEK_API_KEY not set」**：`config/.env` 没配 key。
- **检索结果为空/少**：Semantic Scholar 限流了，等几分钟再试。
