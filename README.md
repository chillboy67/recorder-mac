# Recorder — 本地离线录音转文字

一个在 MacBook 本地运行、**双击即开**的录音转文字 App。以转写准确性为根基，
提供三种模式：

| 模式 | 用途 |
|------|------|
| **通用转写** | 最高精度、忠实原文的语音转文字（中英混合自动识别） |
| **课堂录音** | 空旷/有回声的教室：轻度降噪 + 聚焦主讲人 + 排除旁人闲聊 |
| **雅思口语教官** | 区分教官/考生，标注疑似**读音**、语法、表达问题，生成反馈（绝不修改原文） |

全程离线，不调用任何云 API。

> 实际代码在 [`lecture_intel/`](lecture_intel/) 目录（历史命名，App 名为 Recorder）。
> 早期的 `backend/`（FastAPI）+ `frontend/` 是原型，已被 `lecture_intel/` 取代，
> 见下方"历史"。完整需求见 [REQUIREMENTS.md](REQUIREMENTS.md)。

## 快速开始

```bash
cd lecture_intel

# 1) 准备环境（已有 .venv 可跳过）
uv venv
uv pip install -r requirements.txt

# 2) 打包成双击 App
./make_app.sh
open dist/                       # 把 Recorder.app 拖到 /Applications

# 或直接命令行运行 GUI
.venv/bin/python3 app.py
```

首次启动会下载 Whisper 模型（默认 `large-v3`，约 3GB，仅一次）。之后完全离线。

国内网络建议先用镜像把模型全部下好（放进 App 的本地模型目录，运行时不再联网）：

```bash
cd lecture_intel
.venv/bin/python3 download_models.py          # 全部模型，走 hf-mirror.com 镜像
```

### 命令行（无界面）

```bash
.venv/bin/python3 transcribe.py 录音.m4a                 # 通用
.venv/bin/python3 transcribe.py 课堂.mp3   -m classroom   # 课堂
.venv/bin/python3 transcribe.py 雅思.webm  -m ielts       # 雅思反馈
.venv/bin/python3 transcribe.py a.wav --model large-v3-turbo -f txt -f srt
```

## 准确性是怎么做上去的

这是相比一般转写 App 的核心改进：

1. **整文件交给 Whisper**，而不是先用 VAD 切成独立 30s 小段再逐段转写。
   旧做法在切口处丢词/重复、上下文断裂；Whisper 自带 30s 窗口与跨窗上下文，
   直接喂整段最准。
2. **Apple Silicon 加速**：默认 `mlx-whisper`（M 系列 Metal 加速，接近实时），
   不可用时自动回退 `faster-whisper`（CPU）。
3. **逐词置信度**始终开启 —— 这是雅思模式识别"疑似发音问题"的依据：
   声学模型不确定的词，往往就是读不清/读错/口音偏差的词。
4. **忠实原文**：任何模式都不做 LLM 改写/润色。说错的保留原样，
   这正是诊断价值所在。

## 三种模式细节

- **通用**：加载 → 转写 → 导出（txt/md/srt/json，带时间戳）。
- **课堂**：ffmpeg 降噪（去低频隆隆声 + 自适应降噪 + 响度归一）→ 转写 →
  说话人聚类后只保留时长最长的主讲人 → 导出。
- **雅思**：转写 → 免 token 的 2 说话人分离（Resemblyzer 声纹 + 聚类，
  自动判定教官/考生，UI 可手动对调）→ 分析（低置信度发音点、语法/用词、
  中式表达、教官纠正参考）→ 反馈报告 `*.ielts.md` + 逐字转写。

## 隐私

模型在本机运行；音频与结果只写入输入文件同目录的 `recorder_output/`。
不上传任何数据。

## 历史

`backend/`（FastAPI）+ `frontend/` 是最早的雅思批改原型，依赖较多本地服务
（Whisper CLI、MFA、LanguageTool），已不再是主线。其分析思路（发音/语法/
报告模板）已吸收进 `lecture_intel/core/ielts.py`。保留仅作参考，可忽略。
