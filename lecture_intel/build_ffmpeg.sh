#!/bin/bash
# =============================================================================
# Build a small, audio-only ffmpeg + ffprobe for bundling into Recorder.app.
#
# Most users do not have Homebrew, so asking them to `brew install ffmpeg`
# is the biggest hurdle between downloading the app and using it. The release
# build embeds these two binaries instead (make_app.sh copies them into the
# bundle's bin/ and the launcher puts that directory first on PATH).
#
# Only what the app runs is enabled: the input formats the file picker offers,
# the filters in core/denoise.py, core/repeat_arbitration.py and the recording
# mixer, and the WAV / raw-PCM / null outputs. Nothing is auto-detected, so the
# binaries link against nothing but the system C library, and no GPL component
# is enabled, so they are LGPL-2.1+ (the licence text is copied alongside).
#
# The source tarball is pinned by SHA-256. It is the release signed with the
# FFmpeg release signing key FCF9 86EA 15E6 E293 A564 4F10 B432 2F04 D676 58D8.
#
# Usage:   ./build_ffmpeg.sh            # → dist/ffmpeg/bin/{ffmpeg,ffprobe}
# Env:     FFMPEG_TARBALL=path          # use a local copy instead of downloading
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

VERSION=8.1.2
SHA256=464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c
URL="https://ffmpeg.org/releases/ffmpeg-$VERSION.tar.xz"

OUT="$(pwd)/dist/ffmpeg"
WORK="$(pwd)/dist/ffmpeg-build"
rm -rf "$OUT" "$WORK"
mkdir -p "$OUT/bin" "$WORK"

tarball="${FFMPEG_TARBALL:-$WORK/ffmpeg-$VERSION.tar.xz}"
if [[ -z "${FFMPEG_TARBALL:-}" ]]; then
  echo "→ downloading ffmpeg $VERSION"
  curl -fsSL --retry 5 -o "$tarball" "$URL"
fi
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$tarball" | cut -d' ' -f1)"
else
  actual="$(shasum -a 256 "$tarball" | cut -d' ' -f1)"
fi
if [[ "$actual" != "$SHA256" ]]; then
  echo "SHA-256 mismatch for $tarball: $actual" >&2
  exit 1
fi
tar -xJf "$tarball" -C "$WORK"
src="$WORK/ffmpeg-$VERSION"

jobs="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
# Run on every macOS the app supports (LSMinimumSystemVersion in make_app.sh),
# not just the build machine's.
if [[ "$(uname)" == Darwin ]]; then export MACOSX_DEPLOYMENT_TARGET=12.0; fi

echo "→ configuring"
(cd "$src" && ./configure \
  --prefix="$WORK/install" \
  --disable-everything \
  --disable-autodetect \
  --disable-doc \
  --disable-debug \
  --disable-network \
  --disable-ffplay \
  --disable-avdevice \
  --enable-ffmpeg \
  --enable-ffprobe \
  --enable-protocol=file,pipe \
  --enable-demuxer=mov,mp3,aac,flac,ogg,matroska,wav,w64,aiff,caf \
  --enable-parser=aac,aac_latm,flac,mpegaudio,opus,vorbis \
  --enable-decoder=aac,aac_latm,alac,mp3,mp3float,flac,vorbis,opus,'pcm_*' \
  --enable-encoder=pcm_s16le,pcm_f32le,wrapped_avframe \
  --enable-muxer=wav,pcm_s16le,pcm_f32le,null \
  --enable-filter=aformat,anull,aresample,atrim,asetpts,volume,pan,highpass,afftdn,dynaudnorm,volumedetect,silencedetect,amix \
  >"$WORK/configure.log") || { tail -30 "$WORK/configure.log" >&2; exit 1; }

echo "→ compiling ($jobs jobs)"
make -C "$src" -j"$jobs" >"$WORK/make.log" 2>&1 || { tail -40 "$WORK/make.log" >&2; exit 1; }

cp "$src/ffmpeg" "$src/ffprobe" "$OUT/bin/"
strip "$OUT/bin/ffmpeg" "$OUT/bin/ffprobe"
cp "$src/COPYING.LGPLv2.1" "$OUT/bin/FFMPEG-LICENSE.txt"
printf 'ffmpeg %s, built from %s (LGPL-2.1-or-later).\n' "$VERSION" "$URL" > "$OUT/bin/FFMPEG-SOURCE.txt"
rm -rf "$WORK"

"$OUT/bin/ffmpeg" -hide_banner -version | head -1
echo "✓ $OUT/bin"
