# AURA 界面设计意图（改 `gui/` 前必读）

现役界面是「1b AURA 未来声学」定稿。本文件记录**改动时必须保持**的设计约定——
这些是界面的一部分，不是可选的美化偏好。原先记在仓库根的 `aura_gui/README.md`
（该目录已删除，它是改版的设计稿源本，早已合并进本目录，且代码比现役更旧）。

原始设计稿：`Recorder AURA.dc.html`（未入库）。

## 必须保持

### 1. 录音界面**不设**进度条 / 进度环

录音没有时长上限，进度环会暗示"快结束了"，是错误的心智模型。
录音界面只有 `GlowRing`，它的职责**仅是承载时钟**（`gui/widgets/recording_screen.py:105-106`、
`set_time("00:00:00")` 于同文件 315 行）。

进度环只属于**处理**界面——转写有真实百分比，那里才需要：
`gui/widgets/processing_screen.py` 用 `ProgressRing`。

### 2. 录音的暂停 / 继续是**有条件的**

暂停按钮仅在当前音源能安全暂停时出现：

```python
can_pause = source == "mic" or self._sys_rec.can_pause()
self._pause_btn.setVisible(can_pause)
```

（`gui/widgets/recording_screen.py:145-146`）

麦克风走 `QMediaRecorder.pause()/record()`；电脑内部声音 / 混录**只在
`SystemAudioRecorder.can_pause()` 为真时**才显示暂停按钮，否则自动隐藏——
两路音轨一旦不同步就无法修复，宁可不让用户按。
`recording_screen.py:143` 附近有注释指出旧构建会在暂停信号上直接崩。

### 3. 新控件取色一律走 `theme.current_scheme()`，**不要写死 hex**

深浅两套色表都在 `gui/theme.py`。写死颜色会在另一套主题下失效。
现役调用点（`theme.current_scheme()`）：`gui/widgets/visuals.py:18`、
`common.py:78`、`results_screen.py:184/245/254`、`processing_screen.py:104/135`。

### 4. 时钟与数字用 Space Grotesk

`gui/theme.py:47-48`：

```python
DISPLAY_FAMILIES = ["Space Grotesk", "SF Pro Display", "Helvetica Neue"]
QSS_DISPLAY = '"Space Grotesk", "SF Pro Display", "Helvetica Neue"'
```

装了这个字体更好看，没装会沿链条自动退回，不会坏。
字体（可选，本机已装）：<https://fonts.google.com/specimen/Space+Grotesk>

## 不得改坏的行为

这些是改版时明确要求照搬的既有行为，动界面时不要顺手改掉：

- **偏好持久化键**：`QSettings("LucasLab", "Recorder")`
  （`app.py:64`、`gui/main_window.py:56`、`gui/widgets/home_screen.py:127`）。
  **键名不要改**，否则老用户偏好全部丢失。
- **三种录音来源**：麦克风（Qt Multimedia）/ 电脑内部声音（ScreenCaptureKit）/
  混录（ffmpeg `amix`）。权限提示与"保存到哪"的询问逻辑照旧。
- **菜单栏与快捷键**：`Cmd+O`（`QKeySequence.Open`）、`Cmd+Shift+O`、
  `Cmd+R`（`gui/main_window.py:89/94/103`）。
- **外观跟随系统**；**转写完成发通知**；**窗口几何记忆**。

## 四屏结构

```
main_window.py   顶栏 + 步骤轨(step_rail) + 4 屏 stack
  home_screen.py        空闲页：模式卡片 + 底部设置行（原 settings_panel 已并入）
  recording_screen.py   录音页：GlowRing + WaveBars + PulseDot（无进度环）
  processing_screen.py  处理页：ProgressRing + 玻璃步骤卡
  results_screen.py     结果页：双玻璃面板
  visuals.py, step_rail.py, common.py   自绘控件 / 步骤轨 / 通用件
```

模式选择、模型、导出格式、LLM 增强等偏好在 `home_screen.py`；
进度回调签名沿用 `update_progress(info dict from PipelineWorker)`。
