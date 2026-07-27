# AURA GUI — 安装说明

按 1b「AURA 未来声学」定稿重写的 Recorder 界面(PySide6)。
设计稿:`Recorder AURA.dc.html`。

## 怎么装

1. **备份**:把 `lecture_intel/gui/` 改名为 `lecture_intel/gui_old/`(注意里面的
   `workers/` 还要用,见下一步)。
2. 新建 `lecture_intel/gui/`,把本目录内容复制进去:

   ```
   aura_gui/theme.py                     → lecture_intel/gui/theme.py
   aura_gui/main_window.py               → lecture_intel/gui/main_window.py
   aura_gui/__init__.py                  → lecture_intel/gui/__init__.py
   aura_gui/widgets/*.py                 → lecture_intel/gui/widgets/
   ```

3. **workers 原样保留**:把 `gui_old/workers/` 整个移回 `gui/workers/`。
   (PipelineWorker 未改动。)
4. 运行。入口(`app.py` / `main.py`)不用改:`MainWindow`、
   `theme.apply(app, mode)` 的接口和原来一致。

## 字体(可选但推荐)

时钟和数字用 Space Grotesk。装了更好看,没装会自动退回 SF Pro:
<https://fonts.google.com/specimen/Space+Grotesk> 下载后双击安装 TTF 即可。

## 与旧版的对应关系

| 旧 | 新 |
|---|---|
| theme.py | theme.py(AURA 配色,同名 API) |
| main_window.py(左右分栏) | main_window.py(顶栏 + 步骤轨 + 4 屏 stack) |
| input_panel.py(录音/文件 Tab) | home_screen.py + recording_screen.py |
| settings_panel.py | 并入 home_screen.py(模式卡片 + 底部设置行) |
| progress_panel.py | processing_screen.py(进度环 + 玻璃步骤卡) |
| results_panel.py | results_screen.py(双玻璃面板) |
| — | step_rail.py, visuals.py(新增) |

## 保留的行为

- 三种模式、模型选择、导出格式、LLM 增强等偏好照旧存 QSettings
  (`LucasLab/Recorder`),旧偏好自动生效
- 录音:麦克风(Qt Multimedia)/ 电脑声音(ScreenCaptureKit)/ 混音(ffmpeg amix),
  权限提示、保存询问逻辑照搬
- 菜单栏、快捷键(Cmd+O / Cmd+Shift+O / Cmd+R)、外观跟随系统、完成通知、
  窗口几何记忆均保留

## 设计意图(改动时请保持)

- 录音界面**没有进度条/进度环**——录音无时长上限,柔光环只承载时钟
- 录音支持暂停/继续:麦克风走 QMediaRecorder.pause()/record();
  电脑声音/混音仅当 SystemAudioRecorder 提供 pause()/resume() 时显示
  暂停按钮(否则自动隐藏,避免两路音轨不同步)
- 处理界面才有进度环(转写有真实百分比)
- 深浅色都走 `theme.py` 的两张色表;新控件颜色一律取
  `theme.current_scheme()`,不要写死 hex
