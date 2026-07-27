// SystemAudioRecorder — capture the Mac's own audio output (loopback) to a WAV,
// using ScreenCaptureKit. This taps the digital audio stream directly, so it
// works regardless of output volume or whether headphones are plugged in, and
// needs no virtual-audio driver — only the one-time "Screen Recording" permission.
//
// Usage:  SystemAudioRecorder <output.wav>
//         records until it receives SIGINT/SIGTERM, then finalizes the file.
//         SIGUSR1 pauses (samples are dropped), SIGUSR2 resumes.
// Prints "RECORDING" / "PAUSED" / "RESUMED" / "STOPPED" / "ERROR ..." to stderr
// for the parent process.
import Foundation
import ScreenCaptureKit
import AVFoundation
import CoreMedia

@available(macOS 13.0, *)
final class SysAudioRecorder: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let outURL: URL
    private let q = DispatchQueue(label: "rec.audio")
    // Manual WAV writer (interleaved Int16) — robust against SCK's channel layout.
    private var handle: FileHandle?
    private var sampleRate = 48000
    private var channels = 2
    private var totalFrames = 0
    private var logged = false
    private var paused = false

    init(outputPath: String) { self.outURL = URL(fileURLWithPath: outputPath) }

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            throw NSError(domain: "rec", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "no display available"])
        }
        let filter = SCContentFilter(display: display,
                                     excludingApplications: [], exceptingWindows: [])
        let cfg = SCStreamConfiguration()
        cfg.capturesAudio = true
        cfg.sampleRate = 48000
        cfg.channelCount = 2
        cfg.excludesCurrentProcessAudio = true       // don't capture our own sound
        cfg.width = 2; cfg.height = 2                 // we only want audio
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let s = SCStream(filter: filter, configuration: cfg, delegate: self)
        try s.addStreamOutput(self, type: .audio, sampleHandlerQueue: q)
        try s.addStreamOutput(self, type: .screen, sampleHandlerQueue: q)
        try await s.startCapture()
        self.stream = s
        log("RECORDING")
    }

    func stop() async {
        try? await stream?.stopCapture()
        finalizeWav()
        log("STOPPED")
    }

    func setPaused(_ p: Bool) {
        // Flip the flag on the sample-handler queue so the callback never
        // observes a torn write.
        q.async { self.paused = p }
        log(p ? "PAUSED" : "RESUMED")
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, CMSampleBufferDataIsReady(sampleBuffer) else { return }
        if paused { return }
        guard let pcm = Self.pcmBuffer(from: sampleBuffer),
              let ch = pcm.floatChannelData else { return }
        let frames = Int(pcm.frameLength)
        if frames == 0 { return }
        let nch = Int(pcm.format.channelCount)

        if handle == nil {
            sampleRate = Int(pcm.format.sampleRate)
            channels = nch
            openWav()
        }
        // Deinterleave float → interleaved Int16
        var bytes = Data(capacity: frames * nch * 2)
        for f in 0..<frames {
            for c in 0..<nch {
                var v = ch[c][f]
                v = max(-1.0, min(1.0, v))
                var s = Int16(v * 32767.0).littleEndian
                withUnsafeBytes(of: &s) { bytes.append(contentsOf: $0) }
            }
        }
        handle?.write(bytes)
        totalFrames += frames
    }

    private func openWav() {
        FileManager.default.createFile(atPath: outURL.path, contents: nil)
        handle = try? FileHandle(forWritingTo: outURL)
        handle?.write(Self.wavHeader(sampleRate: sampleRate, channels: channels, frames: 0))
    }

    private func finalizeWav() {
        guard let h = handle else { return }
        // rewrite header with real sizes
        try? h.seek(toOffset: 0)
        h.write(Self.wavHeader(sampleRate: sampleRate, channels: channels, frames: totalFrames))
        try? h.close()
        handle = nil
    }

    static func wavHeader(sampleRate: Int, channels: Int, frames: Int) -> Data {
        let bitsPerSample = 16
        let byteRate = sampleRate * channels * bitsPerSample / 8
        let blockAlign = channels * bitsPerSample / 8
        let dataSize = frames * blockAlign
        var d = Data()
        func u32(_ v: Int) { var x = UInt32(v).littleEndian; withUnsafeBytes(of: &x){ d.append(contentsOf:$0) } }
        func u16(_ v: Int) { var x = UInt16(v).littleEndian; withUnsafeBytes(of: &x){ d.append(contentsOf:$0) } }
        d.append(contentsOf: Array("RIFF".utf8)); u32(36 + dataSize)
        d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(channels)
        u32(sampleRate); u32(byteRate); u16(blockAlign); u16(bitsPerSample)
        d.append(contentsOf: Array("data".utf8)); u32(dataSize)
        return d
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("ERROR \(error.localizedDescription)")
    }

    static func pcmBuffer(from sb: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard let fd = CMSampleBufferGetFormatDescription(sb),
              let asbdP = CMAudioFormatDescriptionGetStreamBasicDescription(fd) else { return nil }
        var asbd = asbdP.pointee
        guard let fmt = AVAudioFormat(streamDescription: &asbd) else { return nil }
        let frames = AVAudioFrameCount(CMSampleBufferGetNumSamples(sb))
        guard frames > 0,
              let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: frames) else { return nil }
        buf.frameLength = frames
        CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sb, at: 0, frameCount: Int32(frames), into: buf.mutableAudioBufferList)
        return buf
    }
}

func log(_ s: String) {
    FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
}

let args = CommandLine.arguments
guard args.count >= 2 else { log("usage: SystemAudioRecorder <out.wav>"); exit(2) }

guard #available(macOS 13.0, *) else { log("ERROR requires macOS 13+"); exit(3) }
let recorder = SysAudioRecorder(outputPath: args[1])

// Stop cleanly on SIGINT/SIGTERM (the parent app sends these to stop recording).
// SIGUSR1/SIGUSR2 pause/resume the capture (parent app's pause button).
signal(SIGINT, SIG_IGN)
signal(SIGTERM, SIG_IGN)
signal(SIGUSR1, SIG_IGN)
signal(SIGUSR2, SIG_IGN)
let stopHandler: () -> Void = {
    Task { await recorder.stop(); exit(0) }
}
let s1 = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
let s2 = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
s1.setEventHandler(handler: stopHandler); s1.resume()
s2.setEventHandler(handler: stopHandler); s2.resume()
let p1 = DispatchSource.makeSignalSource(signal: SIGUSR1, queue: .main)
let p2 = DispatchSource.makeSignalSource(signal: SIGUSR2, queue: .main)
p1.setEventHandler { recorder.setPaused(true) }; p1.resume()
p2.setEventHandler { recorder.setPaused(false) }; p2.resume()

Task {
    do { try await recorder.start() }
    catch { log("START_ERROR \(error.localizedDescription)"); exit(1) }
}
RunLoop.main.run()
