import Foundation
import CoreML

enum CoreMLRunnerError: Error, CustomStringConvertible {
    case missingInputDescriptions
    case missingOutputDescriptions
    case missingOutput(String)
    case unsupportedDataType
    case shapeMismatch(expected: Int, actual: Int)

    var description: String {
        switch self {
        case .missingInputDescriptions:
            return "Core ML model must expose at least two inputs."
        case .missingOutputDescriptions:
            return "Core ML model must expose at least two outputs."
        case .missingOutput(let name):
            return "Missing output feature: \(name)"
        case .unsupportedDataType:
            return "Only float32 MLMultiArray is supported in this example."
        case .shapeMismatch(let expected, let actual):
            return "Array element count mismatch. expected=\(expected), actual=\(actual)"
        }
    }
}

final class CoreMLWTConvRunner {
    private let model: MLModel
    private let baseInputName: String
    private let waveletInputName: String
    private let baseOutputName: String
    private let waveletOutputName: String

    init(modelURL: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let loadURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            loadURL = modelURL
        } else {
            loadURL = try MLModel.compileModel(at: modelURL)
        }
        self.model = try MLModel(contentsOf: loadURL, configuration: config)

        let inputNames = Array(model.modelDescription.inputDescriptionsByName.keys)
        guard inputNames.count >= 2 else {
            throw CoreMLRunnerError.missingInputDescriptions
        }

        let outputNames = Array(model.modelDescription.outputDescriptionsByName.keys)
        guard outputNames.count >= 2 else {
            throw CoreMLRunnerError.missingOutputDescriptions
        }

        if let base = inputNames.first(where: { $0.lowercased().contains("base") }),
           let wavelet = inputNames.first(where: { $0.lowercased().contains("wavelet") }) {
            self.baseInputName = base
            self.waveletInputName = wavelet
        } else {
            self.baseInputName = inputNames[0]
            self.waveletInputName = inputNames[1]
        }

        if let base = outputNames.first(where: { $0.lowercased().contains("base") }),
           let wavelet = outputNames.first(where: { $0.lowercased().contains("wavelet") }) {
            self.baseOutputName = base
            self.waveletOutputName = wavelet
        } else {
            self.baseOutputName = outputNames[0]
            self.waveletOutputName = outputNames[1]
        }
    }

    func predict(
        baseInput: [Float],
        waveletInput: [Float],
        channels: Int,
        height: Int,
        width: Int
    ) throws -> (baseOutput: [Float], waveletOutput: [Float]) {
        let h2 = (height + 1) / 2
        let w2 = (width + 1) / 2

        let baseArray = try MLMultiArray(shape: [1, channels as NSNumber, height as NSNumber, width as NSNumber], dataType: .float32)
        let waveletArray = try MLMultiArray(shape: [1, (channels * 4) as NSNumber, h2 as NSNumber, w2 as NSNumber], dataType: .float32)

        try baseArray.writeContiguous(baseInput)
        try waveletArray.writeContiguous(waveletInput)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            baseInputName: MLFeatureValue(multiArray: baseArray),
            waveletInputName: MLFeatureValue(multiArray: waveletArray)
        ])

        let prediction = try model.prediction(from: provider)

        guard let baseFeature = prediction.featureValue(for: baseOutputName)?.multiArrayValue else {
            throw CoreMLRunnerError.missingOutput(baseOutputName)
        }
        guard let waveletFeature = prediction.featureValue(for: waveletOutputName)?.multiArrayValue else {
            throw CoreMLRunnerError.missingOutput(waveletOutputName)
        }

        let baseOutput = try baseFeature.readContiguous()
        let waveletOutput = try waveletFeature.readContiguous()
        return (baseOutput, waveletOutput)
    }
}

private extension MLMultiArray {
    func writeContiguous(_ values: [Float]) throws {
        guard dataType == .float32 else {
            throw CoreMLRunnerError.unsupportedDataType
        }
        guard values.count == count else {
            throw CoreMLRunnerError.shapeMismatch(expected: count, actual: values.count)
        }

        if isContiguousRowMajor {
            let ptr = dataPointer.bindMemory(to: Float.self, capacity: count)
            values.withUnsafeBufferPointer { src in
                ptr.update(from: src.baseAddress!, count: count)
            }
            return
        }

        let shape = self.shape.map { $0.intValue }
        let strides = self.strides.map { $0.intValue }
        let ptr = dataPointer.bindMemory(to: Float.self, capacity: count)

        for linear in 0..<count {
            var remaining = linear
            var offset = 0
            for dim in stride(from: shape.count - 1, through: 0, by: -1) {
                let idx = remaining % shape[dim]
                remaining /= shape[dim]
                offset += idx * strides[dim]
            }
            ptr[offset] = values[linear]
        }
    }

    func readContiguous() throws -> [Float] {
        guard dataType == .float32 else {
            throw CoreMLRunnerError.unsupportedDataType
        }

        var output = Array<Float>(repeating: 0, count: count)
        let ptr = dataPointer.bindMemory(to: Float.self, capacity: count)

        if isContiguousRowMajor {
            output.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: ptr, count: count)
            }
            return output
        }

        let shape = self.shape.map { $0.intValue }
        let strides = self.strides.map { $0.intValue }

        for linear in 0..<count {
            var remaining = linear
            var offset = 0
            for dim in stride(from: shape.count - 1, through: 0, by: -1) {
                let idx = remaining % shape[dim]
                remaining /= shape[dim]
                offset += idx * strides[dim]
            }
            output[linear] = ptr[offset]
        }

        return output
    }

    var isContiguousRowMajor: Bool {
        let s = shape.map { $0.intValue }
        let st = strides.map { $0.intValue }
        guard s.count == st.count else {
            return false
        }

        var expected = 1
        for dim in stride(from: s.count - 1, through: 0, by: -1) {
            if st[dim] != expected {
                return false
            }
            expected *= s[dim]
        }
        return true
    }
}
