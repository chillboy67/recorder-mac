# Filler eval set / 口头填充评测集

50 条评测语音，测 Whisper 通用模式到底保留多少"说得不流畅"的痕迹。
动机：Whisper 解码时本身会丢一部分语气词（上游 faster-whisper#901、
whisper.cpp#965 报告后均未修复）——"忠实原文"的边界必须用本机流水线 +
自己的声音实测，而不是猜。

## 音频来源（实测采用：TTS 合成）

原设计是照 `filler_set.json` 的 `ground_truth_text` 真人逐字念。实测改用
**TTS 合成**：把每条 `ground_truth_text` 交给本地 TTS 朗读生成
`audio/<id>.wav`（48 kHz/16-bit/mono）。好处是 ground truth 逐字精确、
可复现，不需要真人长时间配音；代价是合成语音不含真实口吃/连读的声学
复杂度，召回率是"干净语音下的上界"。`ground_truth_text` 是脚本，若换用
真实音频，以实际说的为准改 manifest——测量的是"转写 vs 真实说的话"。

真实网上音频（SEP-28k 口吃播客、中文口吃电话录音）用于三模式行为 E2E，
见 `e2e_three_modes.py`，不参与召回计数。

## 实测结果（2026-09-20，large-v3 通用模式，50/50）

- 英文填充词召回 **12/13 = 92%**；中文填充词召回 **9/10 = 90%** → 均 ≥ 50%
  决策门：**不改架构**，数字已回填根 README 的"忠实原文"边界声明。
- 半截话保留 100%（含假阳性放宽）；重复类 80%；混合类 ~80%。
- 丢失形态：吞词（"er"→无、"uh"→无）或近音误写（"嗯"→"恨"、"呃"→"恶"），
  无正文改写、无拼接。

## 跑测量

```bash
cd lecture_intel
.venv/bin/python eval/measure_filler_recall.py                 # 全部已录条目
.venv/bin/python eval/measure_filler_recall.py --model large-v3-turbo
```

缺音频的条目自动跳过；结果写进 `report.md`（分语言×类别的召回表 +
逐条明细）。每个条目的完整引擎输出（含 `original.wav`、`meta.json`、
标注）在 `output/<id>/`，可人工复核。

## 三模式 E2E（真实网上音频）

```bash
.venv/bin/python eval/e2e_three_modes.py <en.wav> <zh.wav>
```

把一段真实口吃/填充词密集的音频分别过 general / classroom / ielts 三种
模式，检查：通用/雅思只标注不动正文；课堂只折叠确认的 ASR 循环伪影、
保留真口吃；`meta.json` 全程留痕。

## CPU / GPU 后端对比

在 `lecture_intel/` 目录用同一段本地音频比较 CPU 与 whisper.cpp 后端；如有
人工校订的参考文本，可同时计算近似误差率：

```bash
.venv/bin/python eval/backend_benchmark.py path/to/sample.wav \
  --engines faster-whisper,whisper.cpp-vulkan \
  --model small --warmups 1 --repeats 3 --reference path/to/reference.txt
```

OpenVINO 用 `whisper.cpp-openvino` 替换 Vulkan。每次会记录请求/实际后端；GPU
请求回退 CPU 时明确标记不合格。JSON 报告含端到端耗时、RTF、中位数和可选 RSS
（安装 `psutil` 后采样），不等同于纯模型推理速度。参考文本分数是英文词级近似，
中文/日文按字符计算；请结合逐字稿人工审听，不要只凭单个分数判断质量。完整说明见
[`docs/GPU_BACKENDS.md`](../docs/GPU_BACKENDS.md)。

## 决策门（跑完按数字执行）

| 结果 | 动作 |
|------|------|
| 英文 filler 召回 **< 50%** | 触发 IELTS 模式 CrisperWhisper 接入 spike（独立后续计划） |
| 中文 filler 召回 **< 50%** | 触发 SenseVoice 对比 spike（独立后续计划） |
| 两者均 **≥ 50%** | 不改架构；把数字填进根 README 的"忠实原文"边界声明即可 |

说明：

- "filler" 类别（um/uh/er/嗯/呃/那个/就是）是门判决据；"repeat" 由
  重复仲裁模块处理（真口吃保留、确认伪影才折叠）；"half"（半截话）的
  匹配允许假阳性（完成的词存活时会误报保留），只作参考，看逐条明细判读。
- spike = 小范围可行性验证，属后续独立计划，不在本评测集范围内。
- 跑完把数字回填：根 README「边界同样要说清」一段与 README.en.md 对应段。
