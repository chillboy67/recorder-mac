# Filler eval set / 口头填充评测集

50 条自录语音，测 Whisper 通用模式到底保留多少"说得不流畅"的痕迹。
动机：Whisper 解码时本身会丢一部分语气词（上游 faster-whisper#901、
whisper.cpp#965 报告后均未修复）——"忠实原文"的边界必须用本机流水线 +
自己的声音实测，而不是猜。

## 录制方法

1. 看 `filler_set.json` 里每条目的 `ground_truth_text`，**照字面念**，
   包括里面的"嗯""uh""no no no no"和破折号（破折号 = 说到一半停住/重新起头）。
2. 每条单独存为 `audio/<id>.wav`（`.m4a/.mp3/.flac` 也行；48 kHz 或
   16 kHz 均可，引擎会归一化）。环境保持你平时使用的样子（同一支麦、
   同一个麦克风模式）。
3. `ground_truth_text` 是脚本，若你现场说得不一样，以实际说的为准改
   manifest——测量的是"转写 vs 你真实说的话"。

## 跑测量

```bash
cd lecture_intel
.venv/bin/python eval/measure_filler_recall.py                 # 全部已录条目
.venv/bin/python eval/measure_filler_recall.py --model large-v3-turbo
```

缺音频的条目自动跳过；结果写进 `report.md`（分语言×类别的召回表 +
逐条明细）。每个条目的完整引擎输出（含 `original.wav`、`meta.json`、
标注）在 `output/<id>/`，可人工复核。

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
