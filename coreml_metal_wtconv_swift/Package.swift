// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "WTConvCoreMLRunner",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "WTConvCoreMLRunner", targets: ["WTConvCoreMLRunner"])
    ],
    targets: [
        .executableTarget(
            name: "WTConvCoreMLRunner",
            resources: [
                .copy("Resources/haar_single.metal"),
                .copy("Resources/haar_inverse.metal")
            ]
        )
    ]
)
