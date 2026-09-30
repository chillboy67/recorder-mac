# 在 Linux 上安装 Recorder

Linux 版是源码引导安装：从 Release 下载的压缩包里是源码，`install_linux.sh` 会在程序目录里建一个
独立的 Python 环境（`.venv/`）并装好依赖。它不是单文件可执行程序，一台全新的系统需要先装几个系统包。

CI 在 Ubuntu 24.04 上按下面的步骤完整跑通：安装、在真实 PipeWire 上录系统声音、用真实模型转写、
启动界面。其他发行版的命令按包名对应给出，没有在 CI 上跑过。

## 1. 装系统包

需要：Python 3.10+（带 venv 模块）、ffmpeg、录系统声音用的 `pw-record` 或 `parec`、Qt 界面的 xcb 光标库。

**Ubuntu / Debian**

```bash
sudo apt install python3 python3-venv ffmpeg pipewire-bin pulseaudio-utils libxcb-cursor0
```

**Fedora**

```bash
sudo dnf install python3 ffmpeg-free pipewire-utils pulseaudio-utils xcb-util-cursor
```

**Arch**

```bash
sudo pacman -S python ffmpeg pipewire libpulse xcb-util-cursor
```

`pipewire-bin` / `pipewire-utils` 提供 `pw-record`，`pulseaudio-utils` / `libpulse` 提供 `parec`；
两个都装最稳，只装其一也能录系统声音。只用麦克风和导入文件的话可以都不装。

## 2. 下载并安装

在 [Releases](https://github.com/chillboy67/recorder-mac/releases/latest) 下载 `Recorder-Linux-*.tar.gz`：

```bash
tar xzf Recorder-Linux-*.tar.gz
cd lecture_intel
bash install_linux.sh --download-cpu-model
```

或者直接用源码：`git clone https://github.com/chillboy67/recorder-mac && cd recorder-mac/lecture_intel`，
再运行同一条 `install_linux.sh`。

安装脚本会：

- 建 `.venv/` 并装依赖，torch 用 CPU 版（整个环境约 2.2 GB；PyPI 默认的 CUDA 版会多出约 3 GB，本程序用不到）；
- 预下载 `small` 语音模型（约 490 MB，`--download-cpu-model` 时）；
- 在「应用程序」菜单里加一个 Recorder 启动项（`~/.local/share/applications/recorder.desktop`）。

模型没下成功不会中断安装，脚本最后会提示，之后重试即可（见第 4 节）。

## 3. 启动

从桌面的应用菜单打开 Recorder，或者：

```bash
./run_linux.sh
```

以后更新：`git pull`（或解压新版覆盖）后重新运行一次 `bash install_linux.sh`。

## 4. 模型与网络

- 模型默认从镜像 `hf-mirror.com` 下载，国内网络一般可以直接用；想走官方源加 `--hf`：

  ```bash
  .venv/bin/python download_models.py --engine cpu small        # 镜像
  .venv/bin/python download_models.py --engine cpu small --hf   # huggingface.co
  ```

- 界面里选「准确」时用 `large-v3`，需要另外下载：`.venv/bin/python download_models.py --engine cpu large-v3`。
- 模型缓存在 `~/.cache/recorder/models/`（设置了 `XDG_CACHE_HOME` 时在它下面）。离线机器可以把另一台机器上的
  这个目录整个拷过来；也可以用 `RECORDER_MODEL_CACHE_DIR=/路径` 指定别的位置。
- 访问不了 `download.pytorch.org` 时，安装脚本会自动改用 PyPI 上的 torch（体积大，但能用）。
  想强制用 PyPI：`RECORDER_TORCH_INDEX_URL= bash install_linux.sh`。

## 5. 录系统声音

Recorder 录的是「默认输出设备」的监听（monitor）：你从扬声器 / 耳机听到的声音。

先确认声音服务在运行：

```bash
pactl info | grep -E "Server Name|Default Sink"
```

能看到 `PulseAudio (on PipeWire …)` 或 `pulseaudio` 以及一个默认输出设备就行。想录别的设备，
在系统声音设置里把它设为默认输出。

想在自己的机器上完整验证一次（放几段测试音，检查录到、暂停期间不录、结尾不丢）：

```bash
.venv/bin/python -m pip install pytest
RECORDER_LIVE_AUDIO_TESTS=1 .venv/bin/python -m pytest tests/test_live_linux_audio.py
```

测试会通过默认输出设备播放约 5 秒的测试音。

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| `Python's venv module is missing` | Debian/Ubuntu 装 `python3-venv`，然后重新运行安装脚本（半途失败的 `.venv` 会自动重建） |
| 启动时报 `Could not load the Qt platform plugin "xcb"` | 装第 1 节里的 xcb 光标库（`libxcb-cursor0` / `xcb-util-cursor`） |
| 转写时提示 `Could not load the CPU Whisper model 'small'` | 模型还没下好：联网后运行第 4 节的下载命令 |
| 「电脑声音」提示「系统音频录制组件不可用」，或录出来没声音 | 装 `pw-record` / `parec`（第 1 节）；用第 5 节确认默认输出设备是你正在听的那个 |
| 读取 / 导出音频出错 | 确认 `ffmpeg -version` 能运行 |
| Wayland 桌面上界面异常 | 目前只在 X11 上验证过，可以先用 `QT_QPA_PLATFORM=xcb ./run_linux.sh` 走 XWayland |

GPU 加速（Vulkan / OpenVINO）是可选项，见 [GPU_BACKENDS.md](GPU_BACKENDS.md)。
