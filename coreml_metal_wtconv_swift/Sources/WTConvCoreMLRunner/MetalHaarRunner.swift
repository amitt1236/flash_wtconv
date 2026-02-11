import Foundation
import Metal

enum MetalHaarError: Error, CustomStringConvertible {
    case unsupportedDevice
    case missingResource(String)
    case functionNotFound(String)
    case pipelineCreation(String)
    case commandFailure(String)
    case invalidInputSize(expected: Int, actual: Int)

    var description: String {
        switch self {
        case .unsupportedDevice:
            return "No Metal device available."
        case .missingResource(let name):
            return "Missing Metal shader resource: \(name)"
        case .functionNotFound(let name):
            return "Metal function not found: \(name)"
        case .pipelineCreation(let message):
            return "Failed to create Metal pipeline: \(message)"
        case .commandFailure(let message):
            return "Metal command failed: \(message)"
        case .invalidInputSize(let expected, let actual):
            return "Invalid input size. expected=\(expected), actual=\(actual)"
        }
    }
}

final class MetalHaarRunner {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let forwardPipeline: MTLComputePipelineState
    private let inversePipeline: MTLComputePipelineState

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalHaarError.unsupportedDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw MetalHaarError.commandFailure("Failed to create command queue")
        }

        let singleURL = try Self.resourceURL(named: "haar_single", ext: "metal")
        let inverseURL = try Self.resourceURL(named: "haar_inverse", ext: "metal")

        let source = try String(contentsOf: singleURL) + "\n" + String(contentsOf: inverseURL)
        let library = try device.makeLibrary(source: source, options: MTLCompileOptions())

        guard let forwardFunction = library.makeFunction(name: "haar2d_forward_kernel") else {
            throw MetalHaarError.functionNotFound("haar2d_forward_kernel")
        }
        guard let inverseFunction = library.makeFunction(name: "haar2d_inverse_kernel") else {
            throw MetalHaarError.functionNotFound("haar2d_inverse_kernel")
        }

        do {
            self.forwardPipeline = try device.makeComputePipelineState(function: forwardFunction)
            self.inversePipeline = try device.makeComputePipelineState(function: inverseFunction)
        } catch {
            throw MetalHaarError.pipelineCreation(error.localizedDescription)
        }

        self.device = device
        self.queue = queue
    }

    func haarForward(
        input: [Float],
        batch: Int,
        channels: Int,
        height: Int,
        width: Int
    ) throws -> [Float] {
        let expected = batch * channels * height * width
        guard input.count == expected else {
            throw MetalHaarError.invalidInputSize(expected: expected, actual: input.count)
        }

        let h2 = (height + 1) / 2
        let w2 = (width + 1) / 2
        let bc = batch * channels

        let outCount = bc * 4 * h2 * w2
        let inBytes = input.count * MemoryLayout<Float>.size
        let outBytes = outCount * MemoryLayout<Float>.size

        guard let inBuffer = device.makeBuffer(bytes: input, length: inBytes, options: .storageModeShared),
              let outBuffer = device.makeBuffer(length: outBytes, options: .storageModeShared),
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw MetalHaarError.commandFailure("Failed to allocate Metal objects for forward pass")
        }

        var c = Int32(channels)
        var h = Int32(height)
        var w = Int32(width)
        var h2v = Int32(h2)
        var w2v = Int32(w2)

        enc.setComputePipelineState(forwardPipeline)
        enc.setBuffer(inBuffer, offset: 0, index: 0)
        enc.setBuffer(outBuffer, offset: 0, index: 1)
        enc.setBytes(&c, length: MemoryLayout<Int32>.size, index: 2)
        enc.setBytes(&h, length: MemoryLayout<Int32>.size, index: 3)
        enc.setBytes(&w, length: MemoryLayout<Int32>.size, index: 4)
        enc.setBytes(&h2v, length: MemoryLayout<Int32>.size, index: 5)
        enc.setBytes(&w2v, length: MemoryLayout<Int32>.size, index: 6)

        let tw = min(16, forwardPipeline.threadExecutionWidth)
        let th = max(1, min(16, forwardPipeline.maxTotalThreadsPerThreadgroup / tw))
        let threads = MTLSize(width: tw, height: th, depth: 1)
        let grid = MTLSize(width: w2, height: h2, depth: bc)

        enc.dispatchThreads(grid, threadsPerThreadgroup: threads)
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        if let error = cmd.error {
            throw MetalHaarError.commandFailure("Forward pass failed: \(error.localizedDescription)")
        }

        let ptr = outBuffer.contents().bindMemory(to: Float.self, capacity: outCount)
        return Array(UnsafeBufferPointer(start: ptr, count: outCount))
    }

    func haarInverse(
        coeffs: [Float],
        batch: Int,
        channels: Int,
        height: Int,
        width: Int
    ) throws -> [Float] {
        let h2 = (height + 1) / 2
        let w2 = (width + 1) / 2
        let bc = batch * channels

        let expected = bc * 4 * h2 * w2
        guard coeffs.count == expected else {
            throw MetalHaarError.invalidInputSize(expected: expected, actual: coeffs.count)
        }

        let outCount = bc * height * width
        let inBytes = coeffs.count * MemoryLayout<Float>.size
        let outBytes = outCount * MemoryLayout<Float>.size

        guard let inBuffer = device.makeBuffer(bytes: coeffs, length: inBytes, options: .storageModeShared),
              let outBuffer = device.makeBuffer(length: outBytes, options: .storageModeShared),
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw MetalHaarError.commandFailure("Failed to allocate Metal objects for inverse pass")
        }

        var h = Int32(height)
        var w = Int32(width)
        var h2v = Int32(h2)
        var w2v = Int32(w2)

        enc.setComputePipelineState(inversePipeline)
        enc.setBuffer(inBuffer, offset: 0, index: 0)
        enc.setBuffer(outBuffer, offset: 0, index: 1)
        enc.setBytes(&h, length: MemoryLayout<Int32>.size, index: 2)
        enc.setBytes(&w, length: MemoryLayout<Int32>.size, index: 3)
        enc.setBytes(&h2v, length: MemoryLayout<Int32>.size, index: 4)
        enc.setBytes(&w2v, length: MemoryLayout<Int32>.size, index: 5)

        let tw = min(16, inversePipeline.threadExecutionWidth)
        let th = max(1, min(16, inversePipeline.maxTotalThreadsPerThreadgroup / tw))
        let threads = MTLSize(width: tw, height: th, depth: 1)
        let grid = MTLSize(width: w2, height: h2, depth: bc)

        enc.dispatchThreads(grid, threadsPerThreadgroup: threads)
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        if let error = cmd.error {
            throw MetalHaarError.commandFailure("Inverse pass failed: \(error.localizedDescription)")
        }

        let ptr = outBuffer.contents().bindMemory(to: Float.self, capacity: outCount)
        return Array(UnsafeBufferPointer(start: ptr, count: outCount))
    }

    private static func resourceURL(named: String, ext: String) throws -> URL {
        #if SWIFT_PACKAGE
        if let url = Bundle.module.url(forResource: named, withExtension: ext) {
            return url
        }
        #endif
        throw MetalHaarError.missingResource("\(named).\(ext)")
    }
}
