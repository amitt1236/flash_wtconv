import Foundation

struct Args {
    let modelPath: String
    let channels: Int
    let height: Int
    let width: Int
    let inputF32Path: String?
    let outputF32Path: String?
}

@inline(__always)
private func now() -> CFAbsoluteTime {
    CFAbsoluteTimeGetCurrent()
}

private func parseArgs() -> Args {
    var modelPath = "Models/WTConvConvModules.mlpackage"
    var channels = 32
    var height = 128
    var width = 128
    var inputF32Path: String?
    var outputF32Path: String?

    var i = 1
    while i < CommandLine.arguments.count {
        let arg = CommandLine.arguments[i]
        switch arg {
        case "--model":
            i += 1
            if i < CommandLine.arguments.count { modelPath = CommandLine.arguments[i] }
        case "--channels":
            i += 1
            if i < CommandLine.arguments.count, let v = Int(CommandLine.arguments[i]) { channels = v }
        case "--height":
            i += 1
            if i < CommandLine.arguments.count, let v = Int(CommandLine.arguments[i]) { height = v }
        case "--width":
            i += 1
            if i < CommandLine.arguments.count, let v = Int(CommandLine.arguments[i]) { width = v }
        case "--input-f32":
            i += 1
            if i < CommandLine.arguments.count { inputF32Path = CommandLine.arguments[i] }
        case "--output-f32":
            i += 1
            if i < CommandLine.arguments.count { outputF32Path = CommandLine.arguments[i] }
        default:
            break
        }
        i += 1
    }

    return Args(
        modelPath: modelPath,
        channels: channels,
        height: height,
        width: width,
        inputF32Path: inputF32Path,
        outputF32Path: outputF32Path
    )
}

private func randomInput(count: Int) -> [Float] {
    var values = Array<Float>(repeating: 0, count: count)
    for i in 0..<count {
        values[i] = Float.random(in: -1.0...1.0)
    }
    return values
}

private func readFloat32Array(path: String, expectedCount: Int) throws -> [Float] {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    let expectedBytes = expectedCount * MemoryLayout<Float>.size
    if data.count != expectedBytes {
        throw NSError(
            domain: "WTConvCoreMLRunner",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "Input byte size mismatch. expected=\(expectedBytes), actual=\(data.count)"]
        )
    }

    return data.withUnsafeBytes { raw in
        let ptr = raw.bindMemory(to: Float.self)
        return Array(ptr)
    }
}

private func writeFloat32Array(path: String, values: [Float]) throws {
    var local = values
    let data = local.withUnsafeMutableBufferPointer { buf in
        Data(bytes: buf.baseAddress!, count: buf.count * MemoryLayout<Float>.size)
    }
    try data.write(to: URL(fileURLWithPath: path))
}

private func add(_ a: [Float], _ b: [Float]) -> [Float] {
    precondition(a.count == b.count)
    var out = Array<Float>(repeating: 0, count: a.count)
    for i in 0..<a.count {
        out[i] = a[i] + b[i]
    }
    return out
}

private func summary(_ v: [Float]) -> String {
    let shown = min(8, v.count)
    let head = v.prefix(shown).map { String(format: "%.5f", $0) }.joined(separator: ", ")
    let checksum = v.reduce(0, +)
    return "first\(shown)=[\(head)] sum=\(String(format: "%.6f", checksum))"
}

do {
    let args = parseArgs()
    let modelURL = URL(fileURLWithPath: args.modelPath)

    print("WTConv CoreML + Metal example")
    print("model=\(modelURL.path)")
    print("shape=(1, \(args.channels), \(args.height), \(args.width))")

    let inputCount = args.channels * args.height * args.width
    let x: [Float]
    if let inputPath = args.inputF32Path {
        x = try readFloat32Array(path: inputPath, expectedCount: inputCount)
        print("input_source=file:\(inputPath)")
    } else {
        x = randomInput(count: inputCount)
        print("input_source=random")
    }

    let metal = try MetalHaarRunner()
    let coreml = try CoreMLWTConvRunner(modelURL: modelURL)

    let t0 = now()
    let coeffs = try metal.haarForward(
        input: x,
        batch: 1,
        channels: args.channels,
        height: args.height,
        width: args.width
    )
    let t1 = now()

    let prediction = try coreml.predict(
        baseInput: x,
        waveletInput: coeffs,
        channels: args.channels,
        height: args.height,
        width: args.width
    )
    let t2 = now()

    let waveletRecon = try metal.haarInverse(
        coeffs: prediction.waveletOutput,
        batch: 1,
        channels: args.channels,
        height: args.height,
        width: args.width
    )
    let t3 = now()

    let y = add(prediction.baseOutput, waveletRecon)
    let t4 = now()

    if let outputPath = args.outputF32Path {
        try writeFloat32Array(path: outputPath, values: y)
        print("output_f32=\(outputPath)")
    }

    print("haar_forward_ms=\(String(format: "%.3f", (t1 - t0) * 1000))")
    print("coreml_ms=\(String(format: "%.3f", (t2 - t1) * 1000))")
    print("haar_inverse_ms=\(String(format: "%.3f", (t3 - t2) * 1000))")
    print("sum_ms=\(String(format: "%.3f", (t4 - t3) * 1000))")
    print("output \(summary(y))")
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
