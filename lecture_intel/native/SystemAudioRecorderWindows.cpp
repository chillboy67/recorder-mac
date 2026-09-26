#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <mmreg.h>
#include <objbase.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "uuid.lib")

namespace {

constexpr REFERENCE_TIME kBufferDuration = 10'000'000;  // One second, in 100 ns units.
constexpr DWORD kCapturePollMilliseconds = 10;
// SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT, spelled out because the SPEAKER_*
// constants live in the driver-only ksmedia.h that the user-mode SDK need not ship.
constexpr DWORD kStereoChannelMask = 0x3;
constexpr std::uint64_t kMaximumWaveDataBytes =
    static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) - 36u;

std::mutex g_logMutex;

void LogLine(const std::string& message) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    std::fwrite(message.data(), 1, message.size(), stderr);
    std::fwrite("\n", 1, 1, stderr);
    std::fflush(stderr);
}

std::string SystemMessage(DWORD code) {
    char buffer[512] = {};
    const DWORD length = FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        code,
        0,
        buffer,
        static_cast<DWORD>(sizeof(buffer)),
        nullptr);

    std::string message;
    if (length != 0) {
        message.assign(buffer, length);
        while (!message.empty() &&
               (message.back() == '\r' || message.back() == '\n' || message.back() == ' ')) {
            message.pop_back();
        }
    }

    char hexadecimal[16] = {};
    std::snprintf(hexadecimal, sizeof(hexadecimal), "0x%08lX", code);
    if (message.empty()) {
        return hexadecimal;
    }
    return std::string(hexadecimal) + " (" + message + ")";
}

std::string HresultMessage(HRESULT result) {
    return SystemMessage(static_cast<DWORD>(result));
}

[[noreturn]] void ThrowHresult(const char* operation, HRESULT result) {
    throw std::runtime_error(std::string(operation) + " failed: " + HresultMessage(result));
}

void CheckHresult(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        ThrowHresult(operation, result);
    }
}

[[noreturn]] void ThrowLastError(const char* operation) {
    throw std::runtime_error(std::string(operation) + " failed: " + SystemMessage(GetLastError()));
}

template <typename T>
class ComPtr {
public:
    ComPtr() noexcept = default;
    ~ComPtr() { Reset(); }

    ComPtr(const ComPtr&) = delete;
    ComPtr& operator=(const ComPtr&) = delete;

    ComPtr(ComPtr&& other) noexcept : pointer_(other.pointer_) {
        other.pointer_ = nullptr;
    }

    ComPtr& operator=(ComPtr&& other) noexcept {
        if (this != &other) {
            Reset();
            pointer_ = other.pointer_;
            other.pointer_ = nullptr;
        }
        return *this;
    }

    T* Get() const noexcept { return pointer_; }
    T* operator->() const noexcept { return pointer_; }

    T** Put() noexcept {
        Reset();
        return &pointer_;
    }

    void Reset(T* value = nullptr) noexcept {
        if (pointer_ != nullptr) {
            pointer_->Release();
        }
        pointer_ = value;
    }

private:
    T* pointer_ = nullptr;
};

class ScopedHandle {
public:
    ScopedHandle() noexcept = default;
    explicit ScopedHandle(HANDLE handle) noexcept : handle_(handle) {}
    ~ScopedHandle() { Reset(); }

    ScopedHandle(const ScopedHandle&) = delete;
    ScopedHandle& operator=(const ScopedHandle&) = delete;

    ScopedHandle(ScopedHandle&& other) noexcept : handle_(other.Release()) {}

    ScopedHandle& operator=(ScopedHandle&& other) noexcept {
        if (this != &other) {
            Reset(other.Release());
        }
        return *this;
    }

    bool IsValid() const noexcept {
        return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE;
    }

    HANDLE Get() const noexcept { return handle_; }

    HANDLE Release() noexcept {
        const HANDLE result = handle_;
        handle_ = nullptr;
        return result;
    }

    void Reset(HANDLE value = nullptr) noexcept {
        if (IsValid()) {
            CloseHandle(handle_);
        }
        handle_ = value;
    }

private:
    HANDLE handle_ = nullptr;
};

class ComApartment {
public:
    ComApartment() {
        result_ = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(result_)) {
            ThrowHresult("CoInitializeEx", result_);
        }
    }

    ~ComApartment() {
        if (SUCCEEDED(result_)) {
            CoUninitialize();
        }
    }

    ComApartment(const ComApartment&) = delete;
    ComApartment& operator=(const ComApartment&) = delete;

private:
    HRESULT result_ = E_FAIL;
};

struct CoTaskMemFormat {
    ~CoTaskMemFormat() {
        if (value != nullptr) {
            CoTaskMemFree(value);
        }
    }

    WAVEFORMATEX* value = nullptr;
};

enum class SampleEncoding {
    PcmInteger,
    IeeeFloat,
};

struct InputFormat {
    SampleEncoding encoding = SampleEncoding::PcmInteger;
    std::uint16_t channels = 0;
    std::uint16_t containerBits = 0;
    std::uint16_t validBits = 0;
    std::uint16_t bytesPerSample = 0;
    std::uint32_t sampleRate = 0;
    std::uint16_t blockAlign = 0;
};

GUID WaveSubformat(WORD formatTag) {
    GUID result = {};
    result.Data1 = formatTag;
    result.Data2 = 0x0000;
    result.Data3 = 0x0010;
    result.Data4[0] = 0x80;
    result.Data4[1] = 0x00;
    result.Data4[2] = 0x00;
    result.Data4[3] = 0xaa;
    result.Data4[4] = 0x00;
    result.Data4[5] = 0x38;
    result.Data4[6] = 0x9b;
    result.Data4[7] = 0x71;
    return result;
}

bool IsWaveSubformat(const GUID& value, WORD formatTag) {
    const GUID expected = WaveSubformat(formatTag);
    return std::memcmp(&value, &expected, sizeof(GUID)) == 0;
}

InputFormat ParseInputFormat(const WAVEFORMATEX* waveFormat) {
    if (waveFormat == nullptr) {
        throw std::runtime_error("audio format is null");
    }
    if (waveFormat->nChannels == 0 || waveFormat->nSamplesPerSec == 0) {
        throw std::runtime_error("audio format has an invalid channel count or sample rate");
    }
    if (waveFormat->wBitsPerSample == 0 || (waveFormat->wBitsPerSample % 8) != 0) {
        throw std::runtime_error("audio format uses an unsupported sample container size");
    }

    WORD effectiveTag = waveFormat->wFormatTag;
    WORD validBits = waveFormat->wBitsPerSample;
    if (waveFormat->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
        if (waveFormat->cbSize <
            sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX)) {
            throw std::runtime_error("WAVE_FORMAT_EXTENSIBLE data is truncated");
        }
        const auto* extensible = reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(waveFormat);
        if (IsWaveSubformat(extensible->SubFormat, WAVE_FORMAT_PCM)) {
            effectiveTag = WAVE_FORMAT_PCM;
        } else if (IsWaveSubformat(extensible->SubFormat, WAVE_FORMAT_IEEE_FLOAT)) {
            effectiveTag = WAVE_FORMAT_IEEE_FLOAT;
        } else {
            throw std::runtime_error("endpoint mix format has an unsupported extensible subtype");
        }
        if (extensible->Samples.wValidBitsPerSample != 0) {
            validBits = extensible->Samples.wValidBitsPerSample;
        }
    }

    InputFormat result;
    result.channels = waveFormat->nChannels;
    result.containerBits = waveFormat->wBitsPerSample;
    result.validBits = validBits;
    result.bytesPerSample = static_cast<std::uint16_t>(waveFormat->wBitsPerSample / 8);
    result.sampleRate = waveFormat->nSamplesPerSec;
    result.blockAlign = waveFormat->nBlockAlign;

    if (result.validBits == 0 || result.validBits > result.containerBits) {
        throw std::runtime_error("audio format has an invalid valid-bit count");
    }

    const std::uint32_t minimumBlockAlign =
        static_cast<std::uint32_t>(result.channels) * result.bytesPerSample;
    if (result.blockAlign < minimumBlockAlign) {
        throw std::runtime_error("audio format has an invalid block alignment");
    }

    if (effectiveTag == WAVE_FORMAT_PCM) {
        if (result.containerBits != 8 && result.containerBits != 16 &&
            result.containerBits != 24 && result.containerBits != 32) {
            throw std::runtime_error("unsupported PCM sample width");
        }
        result.encoding = SampleEncoding::PcmInteger;
    } else if (effectiveTag == WAVE_FORMAT_IEEE_FLOAT) {
        if (result.containerBits != 32 && result.containerBits != 64) {
            throw std::runtime_error("unsupported IEEE-float sample width");
        }
        result.encoding = SampleEncoding::IeeeFloat;
    } else {
        throw std::runtime_error("endpoint mix format is neither PCM nor IEEE float");
    }

    return result;
}

WAVEFORMATEXTENSIBLE PreferredFormat() {
    WAVEFORMATEXTENSIBLE format = {};
    format.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
    format.Format.nChannels = 2;
    format.Format.nSamplesPerSec = 48'000;
    format.Format.wBitsPerSample = 32;
    format.Format.nBlockAlign = static_cast<WORD>(2u * sizeof(float));
    format.Format.nAvgBytesPerSec =
        format.Format.nSamplesPerSec * format.Format.nBlockAlign;
    format.Format.cbSize =
        static_cast<WORD>(sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX));
    format.Samples.wValidBitsPerSample = 32;
    format.dwChannelMask = kStereoChannelMask;
    format.SubFormat = WaveSubformat(WAVE_FORMAT_IEEE_FLOAT);
    return format;
}

void ActivateAudioClient(IMMDevice* device, ComPtr<IAudioClient>& client) {
    CheckHresult(
        device->Activate(
            __uuidof(IAudioClient),
            CLSCTX_ALL,
            nullptr,
            reinterpret_cast<void**>(client.Put())),
        "IMMDevice::Activate(IAudioClient)");
}

void InitializeAudio(
    ComPtr<IAudioClient>& audioClient,
    ComPtr<IAudioCaptureClient>& captureClient,
    InputFormat& inputFormat) {
    ComPtr<IMMDeviceEnumerator> enumerator;
    CheckHresult(
        CoCreateInstance(
            __uuidof(MMDeviceEnumerator),
            nullptr,
            CLSCTX_ALL,
            __uuidof(IMMDeviceEnumerator),
            reinterpret_cast<void**>(enumerator.Put())),
        "CoCreateInstance(MMDeviceEnumerator)");

    ComPtr<IMMDevice> endpoint;
    CheckHresult(
        enumerator->GetDefaultAudioEndpoint(eRender, eConsole, endpoint.Put()),
        "GetDefaultAudioEndpoint(eRender)");

    ComPtr<IAudioClient> preferredClient;
    ActivateAudioClient(endpoint.Get(), preferredClient);
    WAVEFORMATEXTENSIBLE preferredFormat = PreferredFormat();
    const DWORD preferredFlags =
        AUDCLNT_STREAMFLAGS_LOOPBACK |
        AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
        AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
    const HRESULT preferredResult = preferredClient->Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        preferredFlags,
        kBufferDuration,
        0,
        &preferredFormat.Format,
        nullptr);

    if (SUCCEEDED(preferredResult)) {
        inputFormat = ParseInputFormat(&preferredFormat.Format);
        audioClient = std::move(preferredClient);
    } else {
        ComPtr<IAudioClient> fallbackClient;
        ActivateAudioClient(endpoint.Get(), fallbackClient);

        CoTaskMemFormat mixFormat;
        CheckHresult(fallbackClient->GetMixFormat(&mixFormat.value), "IAudioClient::GetMixFormat");
        InputFormat parsedMixFormat = ParseInputFormat(mixFormat.value);
        const HRESULT fallbackResult = fallbackClient->Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            kBufferDuration,
            0,
            mixFormat.value,
            nullptr);
        if (FAILED(fallbackResult)) {
            throw std::runtime_error(
                "48 kHz stereo float initialization failed: " +
                HresultMessage(preferredResult) +
                "; endpoint mix-format fallback failed: " +
                HresultMessage(fallbackResult));
        }

        inputFormat = parsedMixFormat;
        audioClient = std::move(fallbackClient);
    }

    CheckHresult(
        audioClient->GetService(
            __uuidof(IAudioCaptureClient),
            reinterpret_cast<void**>(captureClient.Put())),
        "IAudioClient::GetService(IAudioCaptureClient)");
}

void PutU16(BYTE* destination, std::uint16_t value) {
    destination[0] = static_cast<BYTE>(value & 0xffu);
    destination[1] = static_cast<BYTE>((value >> 8u) & 0xffu);
}

void PutU32(BYTE* destination, std::uint32_t value) {
    destination[0] = static_cast<BYTE>(value & 0xffu);
    destination[1] = static_cast<BYTE>((value >> 8u) & 0xffu);
    destination[2] = static_cast<BYTE>((value >> 16u) & 0xffu);
    destination[3] = static_cast<BYTE>((value >> 24u) & 0xffu);
}

class WaveWriter {
public:
    WaveWriter() = default;

    ~WaveWriter() {
        if (file_.IsValid()) {
            try {
                std::string ignored;
                Finalize(ignored);
            } catch (...) {
                file_.Reset();
            }
        }
    }

    WaveWriter(const WaveWriter&) = delete;
    WaveWriter& operator=(const WaveWriter&) = delete;

    void Open(const wchar_t* outputPath, std::uint32_t sampleRate, std::uint16_t channels) {
        if (sampleRate == 0 || channels == 0) {
            throw std::runtime_error("cannot create WAV with an invalid format");
        }
        const std::uint64_t byteRate =
            static_cast<std::uint64_t>(sampleRate) * channels * sizeof(std::int16_t);
        const std::uint32_t blockAlign =
            static_cast<std::uint32_t>(channels) * sizeof(std::int16_t);
        if (byteRate > std::numeric_limits<std::uint32_t>::max() ||
            blockAlign > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("output WAV format exceeds RIFF limits");
        }

        sampleRate_ = sampleRate;
        channels_ = channels;
        file_.Reset(CreateFileW(
            outputPath,
            GENERIC_WRITE,
            FILE_SHARE_READ,
            nullptr,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr));
        if (!file_.IsValid()) {
            ThrowLastError("CreateFileW(output WAV)");
        }

        const auto header = MakeHeader(0);
        std::string error;
        if (!WriteRaw(header.data(), header.size(), error)) {
            file_.Reset();
            throw std::runtime_error(error);
        }
    }

    bool WriteSamples(const std::vector<std::int16_t>& samples, std::string& error) {
        if (samples.empty()) {
            return true;
        }
        const std::uint64_t byteCount =
            static_cast<std::uint64_t>(samples.size()) * sizeof(std::int16_t);
        if (byteCount > kMaximumWaveDataBytes - dataBytes_) {
            error = "WAV exceeds the 4 GiB RIFF size limit";
            return false;
        }
        if (!WriteRaw(samples.data(), static_cast<std::size_t>(byteCount), error)) {
            return false;
        }
        dataBytes_ += byteCount;
        return true;
    }

    bool Finalize(std::string& error) {
        if (!file_.IsValid()) {
            return error.empty();
        }

        bool success = true;
        LARGE_INTEGER beginning = {};
        if (!SetFilePointerEx(file_.Get(), beginning, nullptr, FILE_BEGIN)) {
            error = "SetFilePointerEx(WAV header) failed: " + SystemMessage(GetLastError());
            success = false;
        } else {
            const auto header = MakeHeader(dataBytes_);
            std::string writeError;
            if (!WriteRaw(header.data(), header.size(), writeError)) {
                error = writeError;
                success = false;
            }
        }

        if (!FlushFileBuffers(file_.Get()) && success) {
            error = "FlushFileBuffers(WAV) failed: " + SystemMessage(GetLastError());
            success = false;
        }
        if (!CloseHandle(file_.Release()) && success) {
            error = "CloseHandle(WAV) failed: " + SystemMessage(GetLastError());
            success = false;
        }
        return success;
    }

private:
    std::array<BYTE, 44> MakeHeader(std::uint64_t dataBytes) const {
        std::array<BYTE, 44> header = {};
        std::memcpy(header.data() + 0, "RIFF", 4);
        PutU32(header.data() + 4, static_cast<std::uint32_t>(36u + dataBytes));
        std::memcpy(header.data() + 8, "WAVE", 4);
        std::memcpy(header.data() + 12, "fmt ", 4);
        PutU32(header.data() + 16, 16);
        PutU16(header.data() + 20, WAVE_FORMAT_PCM);
        PutU16(header.data() + 22, channels_);
        PutU32(header.data() + 24, sampleRate_);
        PutU32(
            header.data() + 28,
            sampleRate_ * static_cast<std::uint32_t>(channels_) *
                static_cast<std::uint32_t>(sizeof(std::int16_t)));
        PutU16(
            header.data() + 32,
            static_cast<std::uint16_t>(channels_ * sizeof(std::int16_t)));
        PutU16(header.data() + 34, 16);
        std::memcpy(header.data() + 36, "data", 4);
        PutU32(header.data() + 40, static_cast<std::uint32_t>(dataBytes));
        return header;
    }

    bool WriteRaw(const void* data, std::size_t byteCount, std::string& error) {
        const BYTE* cursor = static_cast<const BYTE*>(data);
        std::size_t remaining = byteCount;
        while (remaining != 0) {
            const DWORD requested = static_cast<DWORD>(
                remaining > std::numeric_limits<DWORD>::max()
                    ? std::numeric_limits<DWORD>::max()
                    : remaining);
            DWORD written = 0;
            if (!WriteFile(file_.Get(), cursor, requested, &written, nullptr)) {
                error = "WriteFile(WAV) failed: " + SystemMessage(GetLastError());
                return false;
            }
            if (written == 0) {
                error = "WriteFile(WAV) wrote zero bytes";
                return false;
            }
            cursor += written;
            remaining -= written;
        }
        return true;
    }

    ScopedHandle file_;
    std::uint32_t sampleRate_ = 0;
    std::uint16_t channels_ = 0;
    std::uint64_t dataBytes_ = 0;
};

std::int64_t ReadSignedLittleEndian(const BYTE* source, std::uint16_t byteCount) {
    std::uint64_t value = 0;
    for (std::uint16_t index = 0; index < byteCount; ++index) {
        value |= static_cast<std::uint64_t>(source[index]) << (index * 8u);
    }
    const std::uint16_t bitCount = static_cast<std::uint16_t>(byteCount * 8u);
    if (bitCount < 64 && (value & (std::uint64_t{1} << (bitCount - 1u))) != 0) {
        value |= (~std::uint64_t{0}) << bitCount;
    }
    return static_cast<std::int64_t>(value);
}

std::int16_t IntegerToPcm16(const BYTE* source, const InputFormat& format) {
    if (format.containerBits == 8) {
        return static_cast<std::int16_t>(
            (static_cast<int>(source[0]) - 128) << 8);
    }

    std::int64_t value = ReadSignedLittleEndian(source, format.bytesPerSample);
    if (format.containerBits > 16) {
        value >>= (format.containerBits - 16u);
    } else if (format.containerBits < 16) {
        value <<= (16u - format.containerBits);
    }

    if (value > std::numeric_limits<std::int16_t>::max()) {
        value = std::numeric_limits<std::int16_t>::max();
    } else if (value < std::numeric_limits<std::int16_t>::min()) {
        value = std::numeric_limits<std::int16_t>::min();
    }
    return static_cast<std::int16_t>(value);
}

std::int16_t FloatToPcm16(double value) {
    if (!std::isfinite(value)) {
        return 0;
    }
    if (value >= 1.0) {
        return std::numeric_limits<std::int16_t>::max();
    }
    if (value <= -1.0) {
        return std::numeric_limits<std::int16_t>::min();
    }

    const double scale = value >= 0.0 ? 32767.0 : 32768.0;
    return static_cast<std::int16_t>(std::lround(value * scale));
}

std::int16_t SampleToPcm16(const BYTE* source, const InputFormat& format) {
    if (format.encoding == SampleEncoding::PcmInteger) {
        return IntegerToPcm16(source, format);
    }

    if (format.containerBits == 32) {
        float value = 0.0f;
        std::memcpy(&value, source, sizeof(value));
        return FloatToPcm16(value);
    }

    double value = 0.0;
    std::memcpy(&value, source, sizeof(value));
    return FloatToPcm16(value);
}

bool ConvertPacketToPcm16(
    const BYTE* packet,
    UINT32 frameCount,
    DWORD flags,
    const InputFormat& format,
    std::vector<std::int16_t>& output,
    std::string& error) {
    if (frameCount >
        std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(format.channels)) {
        error = "captured packet is too large";
        return false;
    }

    const std::size_t sampleCount =
        static_cast<std::size_t>(frameCount) * format.channels;
    output.assign(sampleCount, 0);
    if ((flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0) {
        return true;
    }
    if (packet == nullptr) {
        error = "WASAPI returned a null non-silent packet";
        return false;
    }

    for (UINT32 frame = 0; frame < frameCount; ++frame) {
        const BYTE* frameData =
            packet + static_cast<std::size_t>(frame) * format.blockAlign;
        for (std::uint16_t channel = 0; channel < format.channels; ++channel) {
            const BYTE* sampleData =
                frameData + static_cast<std::size_t>(channel) * format.bytesPerSample;
            output[static_cast<std::size_t>(frame) * format.channels + channel] =
                SampleToPcm16(sampleData, format);
        }
    }
    return true;
}

class ControlReader {
public:
    ControlReader() {
        standardInput_ = GetStdHandle(STD_INPUT_HANDLE);
        if (standardInput_ == nullptr || standardInput_ == INVALID_HANDLE_VALUE) {
            ThrowLastError("GetStdHandle(STD_INPUT_HANDLE)");
        }

        startEvent_.Reset(CreateEventW(nullptr, TRUE, FALSE, nullptr));
        if (!startEvent_.IsValid()) {
            ThrowLastError("CreateEventW(control start)");
        }
        stopEvent_.Reset(CreateEventW(nullptr, TRUE, FALSE, nullptr));
        if (!stopEvent_.IsValid()) {
            ThrowLastError("CreateEventW(control stop)");
        }

        thread_.Reset(CreateThread(nullptr, 0, &ControlReader::ThreadEntry, this, 0, nullptr));
        if (!thread_.IsValid()) {
            ThrowLastError("CreateThread(stdin control)");
        }
    }

    ~ControlReader() { Shutdown(); }

    ControlReader(const ControlReader&) = delete;
    ControlReader& operator=(const ControlReader&) = delete;

    bool Begin(std::string& error) {
        if (!SetEvent(startEvent_.Get())) {
            error = "SetEvent(control start) failed: " + SystemMessage(GetLastError());
            return false;
        }
        return true;
    }

    DWORD WaitForStop(DWORD milliseconds) const noexcept {
        return WaitForSingleObject(stopEvent_.Get(), milliseconds);
    }

    bool IsPaused() const noexcept { return paused_.load(std::memory_order_acquire); }

    bool IsStopRequested() const noexcept {
        return stopRequested_.load(std::memory_order_acquire);
    }

    std::string Error() const {
        std::lock_guard<std::mutex> lock(errorMutex_);
        return error_;
    }

    void Shutdown() noexcept {
        bool expected = false;
        if (!shutdownStarted_.compare_exchange_strong(expected, true)) {
            return;
        }

        shuttingDown_.store(true, std::memory_order_release);
        if (startEvent_.IsValid()) {
            SetEvent(startEvent_.Get());
        }
        if (thread_.IsValid()) {
            CancelIoEx(standardInput_, nullptr);
            CancelSynchronousIo(thread_.Get());
            WaitForSingleObject(thread_.Get(), INFINITE);
            thread_.Reset();
        }
    }

private:
    static DWORD WINAPI ThreadEntry(void* context) noexcept {
        auto* reader = static_cast<ControlReader*>(context);
        try {
            reader->ThreadMain();
        } catch (const std::exception& exception) {
            reader->TrySetError(exception.what());
        } catch (...) {
            reader->TrySetError("unexpected stdin control failure");
        }
        return 0;
    }

    void ThreadMain() {
        const DWORD startWait = WaitForSingleObject(startEvent_.Get(), INFINITE);
        if (startWait != WAIT_OBJECT_0 || shuttingDown_.load(std::memory_order_acquire)) {
            return;
        }

        std::string pending;
        char buffer[256] = {};
        while (!shuttingDown_.load(std::memory_order_acquire)) {
            DWORD bytesRead = 0;
            const BOOL readSucceeded = ReadFile(
                standardInput_, buffer, static_cast<DWORD>(sizeof(buffer)), &bytesRead, nullptr);
            if (!readSucceeded) {
                const DWORD error = GetLastError();
                if (shuttingDown_.load(std::memory_order_acquire) ||
                    error == ERROR_OPERATION_ABORTED) {
                    return;
                }
                SetError("stdin read failed: " + SystemMessage(error));
                return;
            }
            if (bytesRead == 0) {
                if (!pending.empty()) {
                    ProcessLine(pending);
                }
                RequestStop();
                return;
            }

            pending.append(buffer, bytesRead);
            if (pending.size() > 4096) {
                SetError("stdin control line exceeds 4096 bytes");
                return;
            }

            std::size_t newline = 0;
            while ((newline = pending.find('\n')) != std::string::npos) {
                std::string line = pending.substr(0, newline);
                pending.erase(0, newline + 1);
                ProcessLine(line);
                if (stopRequested_.load(std::memory_order_acquire)) {
                    return;
                }
            }
        }
    }

    void ProcessLine(std::string line) {
        while (!line.empty() &&
               (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        std::size_t beginning = 0;
        while (beginning < line.size() &&
               (line[beginning] == ' ' || line[beginning] == '\t')) {
            ++beginning;
        }
        if (beginning != 0) {
            line.erase(0, beginning);
        }

        if (line == "PAUSE") {
            paused_.store(true, std::memory_order_release);
            LogLine("PAUSED");
        } else if (line == "RESUME") {
            paused_.store(false, std::memory_order_release);
            LogLine("RESUMED");
        } else if (line == "STOP") {
            RequestStop();
        }
    }

    void RequestStop() noexcept {
        stopRequested_.store(true, std::memory_order_release);
        SetEvent(stopEvent_.Get());
    }

    void SetError(const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(errorMutex_);
            error_ = message;
        }
        RequestStop();
    }

    void TrySetError(const std::string& message) noexcept {
        try {
            SetError(message);
        } catch (...) {
            RequestStop();
        }
    }

    HANDLE standardInput_ = INVALID_HANDLE_VALUE;
    ScopedHandle startEvent_;
    ScopedHandle stopEvent_;
    ScopedHandle thread_;
    std::atomic<bool> paused_{false};
    std::atomic<bool> stopRequested_{false};
    std::atomic<bool> shuttingDown_{false};
    std::atomic<bool> shutdownStarted_{false};
    mutable std::mutex errorMutex_;
    std::string error_;
};

bool DrainPackets(
    IAudioCaptureClient* captureClient,
    const InputFormat& inputFormat,
    const ControlReader& controls,
    WaveWriter& writer,
    std::vector<std::int16_t>& conversionBuffer,
    std::string& error) {
    while (true) {
        if (controls.IsStopRequested()) {
            return true;
        }

        UINT32 packetFrames = 0;
        const HRESULT nextPacketResult = captureClient->GetNextPacketSize(&packetFrames);
        if (FAILED(nextPacketResult)) {
            error = "IAudioCaptureClient::GetNextPacketSize failed: " +
                HresultMessage(nextPacketResult);
            return false;
        }
        if (packetFrames == 0) {
            return true;
        }

        BYTE* packet = nullptr;
        UINT32 frameCount = 0;
        DWORD flags = 0;
        const HRESULT getBufferResult = captureClient->GetBuffer(
            &packet, &frameCount, &flags, nullptr, nullptr);
        if (FAILED(getBufferResult)) {
            error = "IAudioCaptureClient::GetBuffer failed: " +
                HresultMessage(getBufferResult);
            return false;
        }

        bool packetSucceeded = true;
        std::string packetError;
        try {
            if (!controls.IsPaused()) {
                packetSucceeded = ConvertPacketToPcm16(
                    packet,
                    frameCount,
                    flags,
                    inputFormat,
                    conversionBuffer,
                    packetError);
                if (packetSucceeded) {
                    packetSucceeded = writer.WriteSamples(conversionBuffer, packetError);
                }
            }
        } catch (const std::exception& exception) {
            packetSucceeded = false;
            packetError = exception.what();
        } catch (...) {
            packetSucceeded = false;
            packetError = "unexpected packet conversion failure";
        }

        const HRESULT releaseResult = captureClient->ReleaseBuffer(frameCount);
        if (FAILED(releaseResult)) {
            error = "IAudioCaptureClient::ReleaseBuffer failed: " +
                HresultMessage(releaseResult);
            return false;
        }
        if (!packetSucceeded) {
            error = packetError;
            return false;
        }
    }
}

void AppendError(std::string& destination, const std::string& additional) {
    if (additional.empty()) {
        return;
    }
    if (!destination.empty()) {
        destination += "; ";
    }
    destination += additional;
}

int RecordSystemAudio(const wchar_t* outputPath) {
    ComApartment comApartment;

    ComPtr<IAudioClient> audioClient;
    ComPtr<IAudioCaptureClient> captureClient;
    InputFormat inputFormat;
    InitializeAudio(audioClient, captureClient, inputFormat);

    WaveWriter writer;
    writer.Open(outputPath, inputFormat.sampleRate, inputFormat.channels);

    ControlReader controls;
    CheckHresult(audioClient->Start(), "IAudioClient::Start");
    LogLine("RECORDING");

    std::string runError;
    if (!controls.Begin(runError)) {
        // The cleanup path below still stops WASAPI and finalizes the WAV.
    } else {
        std::vector<std::int16_t> conversionBuffer;
        while (runError.empty()) {
            const DWORD waitResult = controls.WaitForStop(kCapturePollMilliseconds);
            if (waitResult == WAIT_OBJECT_0) {
                break;
            }
            if (waitResult != WAIT_TIMEOUT) {
                runError = "WaitForSingleObject(control) failed: " +
                    SystemMessage(GetLastError());
                break;
            }

            if (!DrainPackets(
                    captureClient.Get(),
                    inputFormat,
                    controls,
                    writer,
                    conversionBuffer,
                    runError)) {
                break;
            }
        }
    }

    const HRESULT stopResult = audioClient->Stop();
    controls.Shutdown();

    AppendError(runError, controls.Error());
    if (FAILED(stopResult)) {
        AppendError(
            runError,
            "IAudioClient::Stop failed: " + HresultMessage(stopResult));
    }

    std::string finalizeError;
    try {
        if (!writer.Finalize(finalizeError)) {
            AppendError(runError, finalizeError);
        }
    } catch (const std::exception& exception) {
        AppendError(runError, std::string("WAV finalization failed: ") + exception.what());
    } catch (...) {
        AppendError(runError, "WAV finalization failed unexpectedly");
    }

    if (!runError.empty()) {
        LogLine("ERROR " + runError);
        return 1;
    }

    LogLine("STOPPED");
    return 0;
}

void PrintCapabilities() {
    const char capabilities[] = "STDIN_CONTROL_V1\nPAUSE\nRESUME\nSTOP\n";
    std::fwrite(capabilities, 1, sizeof(capabilities) - 1, stdout);
    std::fflush(stdout);
}

}  // namespace

int wmain(int argumentCount, wchar_t* arguments[]) {
    if (argumentCount == 2 && std::wcscmp(arguments[1], L"--capabilities") == 0) {
        PrintCapabilities();
        return 0;
    }
    if (argumentCount != 2 || arguments[1] == nullptr || arguments[1][0] == L'\0') {
        LogLine("START_ERROR usage: system_audio_recorder.exe <output.wav>");
        return 2;
    }

    try {
        return RecordSystemAudio(arguments[1]);
    } catch (const std::exception& exception) {
        LogLine(std::string("START_ERROR ") + exception.what());
        return 1;
    } catch (...) {
        LogLine("START_ERROR unexpected failure");
        return 1;
    }
}
