# 硬件与本地大模型指南（Ollama）

> 仓库首页 [README.md](../../README.md) 只作简介；**选多大模型、怎么装、不同内存怎么配，以本文为准。**

本文说明 Recorder 的 **可选「本地大模型增强」**：选什么模型、占多少内存、不同电脑怎么配。  
**与 Whisper 语音识别模型无关**（Whisper 见 `download_models.py` 与界面「识别模型」）。

---

## 1. 先分清两层模型

| 层 | 作用 | 下载方式 | 是否进 GitHub |
|----|------|----------|----------------|
| **ASR（Whisper）** | 录音 → 文字 | `download_models.py` / 首次自动 | **否**（本机缓存） |
| **LLM（Ollama）** | 校对错字、课堂总结、雅思点评 | `ollama pull` / 国内镜像 | **否**（体积数 GB，每人自备） |

- 仓库里只写 **模型名字与安装命令**，**绝不提交模型权重**。
- 未安装 Ollama 或未 pull 模型时：界面可勾选增强，运行时 **自动回退离线规则**，不影响转写。

---

## 2. 推荐双模型策略（亚洲 / 欧洲）

| 语系 / 场景 | 推荐家族 | 典型用途 |
|-------------|---------|----------|
| **亚洲**（中 / 日 / 韩等）+ 中文报告 | **Qwen** | 中文总结、中文雅思诊断说明、CJK 教官用语 |
| **欧洲**（英 / 法 / 西 / 德 / 意等） | **Mistral** | 英文雅思诊断、欧语理解与表达 |

原则：

1. **磁盘上可装两只，运行时只加载一只**（按语言路由）。
2. **不要**在 16–32GB 机器上同时常驻 Qwen + Mistral + Whisper large-v3。
3. 推荐流水线：`转写（Whisper）→ 再调用 LLM`，而不是峰值叠满。

> 当前代码默认仍是 `llama3.1:8b`（英）+ `qwen-zh:7b`（中）。  
> 下表中的 **Mistral / qwen2.5** 为 **推荐升级选型**；改默认名需同步改代码（见文末）。

---

## 3. 本机实测环境（维护者参考）

在一份典型开发机上测得（可按你的机器对照）：

| 项 | 值 |
|----|-----|
| 机型 | MacBook Air，**Apple M5** |
| 统一内存 | **16 GB** |
| GPU | 集成 8 核 Metal |
| Ollama | 已安装时可 `ollama list` 查看 |
| 磁盘 | 模型建议预留 **≥15 GB** 空闲（双 7B + Whisper） |

### 16GB Apple Silicon 适合什么？

| 角色 | **推荐 Ollama 标签** | 约占用（Q4 级） | 说明 |
|------|----------------------|-----------------|------|
| 亚洲 / 中文 | **`qwen2.5:7b`** 或已有的 **`qwen-zh:7b`** | 盘约 4.7GB，运行约 5.5–7GB | **默认首选**；与现有中文 prompt 匹配 |
| 欧洲 / 英文 | **`mistral`（7B）** | 盘约 4.4GB，运行约 5–6.5GB | **16GB 默认欧语**；比 Nemo 更省内存 |
| 欧语想更强 | `mistral-nemo`（12B） | 盘约 7.1GB，运行约 8–10GB | **仅建议：转写结束后再开**；勿与 large-v3 峰值叠满 |
| 不推荐作默认 | `qwen2.5:14b`、Mistral Small 22/24B、Mixtral | ≥9GB / ≥13GB | 16GB + Whisper 易交换卡顿 |

**Whisper 搭配建议（16GB）：**

- 常开 AI 增强 → 识别模型优先 **`large-v3-turbo`** 或 **`small`**
- 追求最准且可关 AI → 可用 **`large-v3`**，LLM 等转写完成再跑

你若已安装 `qwen-zh:7b` + `llama3.1:8b`：**可以继续用**；升级路径见 §6。

---

## 4. 不同电脑配置选型表

内存指 **整机统一内存 / 系统 RAM**（Apple Silicon）或 **GPU 显存预算**（独显本，见下）。  
下载体积为 Ollama 库 **常见 Q4 级** 量级，以 `ollama pull` 实际显示为准。

### 4.1 按内存档位（Apple Silicon 优先）

| 内存档位 | 亚洲（Qwen） | 欧洲（Mistral） | 与 Whisper 同机 | 备注 |
|----------|--------------|-----------------|-----------------|------|
| **8 GB** | `qwen2.5:3b` | `mistral`（仅 turbo/small 转写时） | **严格串行**；尽量小模型 | LLM 可选；large-v3 不建议再叠 7B |
| **16 GB** | **`qwen2.5:7b`** / `qwen3:8b` | **`mistral`（7B）** | 7B + turbo 可；large-v3 时 **串行** | **本项目目标机默认档** |
| **24 GB** | `qwen2.5:14b` 或 7B | **`mistral-nemo`** | turbo 常可共存；14B+large-v3 宜串行 | 双策略甜点档 |
| **32 GB** | `qwen2.5:14b` | `mistral-nemo`；可试 Small 22B（串行） | 14B + turbo 从容 | 质量明显好于 7B |
| **48–64 GB+** | `qwen2.5:32b` / Qwen3 30B 级 | `mistral-small:24b`；Mixtral 可选 | 大模型 + Whisper 可共存 | 工作站向 |

### 4.2 模型体积速查（约）

| Ollama 标签 | 参数量级 | 下载约 | 适合内存起点* |
|-------------|---------|--------|----------------|
| `qwen2.5:3b` | 3B | ~1.9 GB | 8 GB |
| `qwen2.5:7b` | 7B | ~4.7 GB | **16 GB** |
| `qwen2.5:14b` | 14B | ~9 GB | 24 GB |
| `qwen2.5:32b` | 32B | ~20 GB | 64 GB |
| `mistral` / `mistral:7b` | 7B | ~4.4 GB | **16 GB** |
| `mistral-nemo` | 12B | ~7.1 GB | 24 GB（16GB 仅串行试） |
| `mistral-small:24b` | 24B | ~14 GB | 32–48 GB |
| `mixtral`（8x7B） | MoE | ~26 GB | 64 GB+ |

\*「适合起点」= 还要跑 macOS + Recorder GUI + Whisper 时的 **默认推荐**，不是「绝对跑不动」。

### 4.3 独显 Windows / Linux 笔记本（简要）

按 **可用显存** 选型（Whisper 与 LLM 若同卡，预算共享）：

| 显存 | 建议 |
|------|------|
| 6–8 GB | 仅 7B Q4 一只；Whisper 用 small/turbo |
| 12 GB | 7B 舒适；Nemo / 14B 紧 |
| 16–24 GB | 14B 或 Nemo；Small 22B 可试 |
| 24 GB+ | 对齐上表 32GB 档 |

---

## 5. 运行时注意（项目实际）

1. **一次只用一个 LLM**：路由选中 Qwen 或 Mistral 之一，不要两只同时 `keep_alive` 很久。  
2. **长课堂**：`llm.py` 会分块调用；中间内存峰值 ≈ 单模型 + 上下文（建议 `num_ctx` 不要盲目开到 32k+）。  
3. **雅思模式**：考生英文分析可用欧洲模型；**报告若要求中文**，亚洲模型（Qwen）通常更稳。  
4. **失败降级**：Ollama 没开 / 模型名对不上 → 警告并走离线启发式（与 `engine.py` 一致）。  
5. **隐私**：请求打到本机 `http://127.0.0.1:11434`，默认不出网。

---

## 6. 安装命令

### 6.1 安装 Ollama（macOS）

```bash
brew install ollama
brew services start ollama
# 或从 https://ollama.com 安装 App
```

### 6.2 推荐：16GB 默认双模型（亚洲 Qwen + 欧洲 Mistral）

**国际网络（Ollama 官方库）：**

```bash
ollama pull qwen2.5:7b    # 亚洲 / 中文报告  ~4.7GB
ollama pull mistral       # 欧洲 / 英文       ~4.4GB
ollama list
```

**国内网络：** 官方 registry 可能较慢，可用 ModelScope 等镜像 pull 后 `ollama cp` / `ollama create` 起别名，与下方「现状模型」类似。

### 6.3 24GB+ 增强档

```bash
ollama pull qwen2.5:14b
ollama pull mistral-nemo
```

### 6.4 现状代码默认（仍可用）

当前引擎默认名：

| 参数 | 默认 | 说明 |
|------|------|------|
| `chinese_model` | `qwen-zh:7b` | 社区 abliterated Qwen2.5-7B（需自建 tag） |
| `llm_model` | `llama3.1:8b` | Meta Llama 3.1 8B |

历史安装方式（ModelScope，见 `lecture_intel/README.md` 亦可）：

```bash
# 英文（Llama）— 可被 mistral 替代
ollama pull modelscope.cn/LLM-Research/Meta-Llama-3.1-8B-Instruct-GGUF
ollama cp modelscope.cn/LLM-Research/Meta-Llama-3.1-8B-Instruct-GGUF llama3.1:8b

# 中文 qwen-zh:7b — 需 GGUF + Modelfile（体积大，勿提交仓库）
```

若已 pull 官方 `qwen2.5:7b`，可临时：

```bash
ollama cp qwen2.5:7b qwen-zh:7b   # 仅当代码仍写死 qwen-zh:7b 时
```

### 6.5 校验

```bash
ollama list
curl -s http://127.0.0.1:11434/api/tags | head
```

App 内勾选「本地大模型增强」→ 跑一条短录音；若模型缺失，日志/界面应提示回退离线处理。

---

## 7. 各模式里 LLM 做什么

| 模式 | LLM 增强（可选） | 原则 |
|------|------------------|------|
| 通用 | 标点 / 明显识别错字整理 | **不改写原意、不翻译** |
| 课堂 | 校对 + 重点总结 | 可据上下文修术语听写 |
| 雅思 | 语法 / 搭配 / 地道度点评 | **考生原文逐字保留**；只出问题列表 |

关掉增强：全部离线规则（置信度发音、LanguageTool/内置语法等）。

---

## 8. 与代码的对应（开发者）

| 文件 | 作用 |
|------|------|
| `core/llm.py` | Ollama 调用、`DEFAULT_MODEL`、`resolve_model` |
| `core/engine.py` | `llm_model` / `chinese_model` 参数与按 `asr.language` 路由 |
| `core/runner.py` | 子进程透传 settings 默认名 |
| `gui/widgets/home_screen.py` | `get_settings()` 中的硬编码模型名 |

若产品默认改为 **Qwen + Mistral**，需同步修改上述默认字符串，并扩展路由（例如 `ja/ko` → Qwen，`fr/es/de` → Mistral），**不仅改本文档**。

---

## 9. 快速决策树

```
内存 ≤8GB  → 小模型 3–4B 或关闭 LLM；Whisper 用 small
内存 16GB  → Qwen 7B + Mistral 7B；串行；Whisper 优先 turbo
内存 24–32GB → Qwen 14B + Mistral Nemo
内存 ≥64GB → Qwen 32B 级 + Mistral Small 24B
只要中文报告稳 → 优先 Qwen，不要只靠欧语模型
只要法西德教官 → Mistral / Nemo 优先
模型文件       → 只 ollama pull，永不 git add
```

---

## 10. 参考来源（量级）

- [Ollama 模型库](https://ollama.com/library)（`qwen2.5` / `mistral` / `mistral-nemo` 等标签与体积）
- 项目目标机：`REQUIREMENTS.md` 中 M 系列 / 16GB 流畅运行设定
- 统一内存机器上 GPU 可用比例通常低于标称 RAM，**选型宁小勿大**

*文档版本：与双模型（亚洲 Qwen / 欧洲 Mistral）选型说明同步；具体 tag 以 Ollama 库为准。*
