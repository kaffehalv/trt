#include "NvInferPlugin.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace nvinfer1;

// Simple logger
class Logger : public ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) { // filter out INFO/VERBOSE
      std::cout << "[TRT] " << static_cast<int>(severity) << ": " << msg
                << '\n';
    }
  }
};

static Logger gLogger;

// Size in bytes of one element of a given DataType
inline std::size_t elementSize(DataType t) {
  switch (t) {
  case DataType::kFLOAT:
    return 4;
  case DataType::kHALF:
    return 2;
  case DataType::kINT8:
    return 1;
  case DataType::kINT32:
    return 4;
  case DataType::kINT64:
    return 8;
  case DataType::kBOOL:
    return 1;
  default:
    throw std::runtime_error("Unsupported TensorRT DataType");
  }
}

inline int64_t volume(const Dims &d) {
  int64_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) {
    v *= d.d[i];
  }
  return v;
}

struct InputInfo {
  std::string name;
  Dims dims;         // original ONNX dims (may have -1 in batch)
  bool dynamicBatch; // true if dims.d[0] < 0
};

// Parse ONNX once and get the input info + builder + network
bool parseOnnx(const std::string &onnxPath, std::unique_ptr<IBuilder> &builder,
               std::unique_ptr<INetworkDefinition> &network,
               std::unique_ptr<nvonnxparser::IParser> &parser,
               InputInfo &inputInfo) {
  builder.reset(createInferBuilder(gLogger));
  if (!builder) {
    std::cerr << "Failed to create TensorRT builder\n";
    return false;
  }

  network.reset(builder->createNetworkV2(0U));
  if (!network) {
    std::cerr << "Failed to create network\n";
    return false;
  }

  parser.reset(nvonnxparser::createParser(*network, gLogger));
  if (!parser) {
    std::cerr << "Failed to create ONNX parser\n";
    return false;
  }

  if (!parser->parseFromFile(onnxPath.c_str(),
                             static_cast<int>(ILogger::Severity::kWARNING))) {
    std::cerr << "Failed to parse ONNX model: " << onnxPath << '\n';
    return false;
  }

  const int nbInputs = network->getNbInputs();
  if (nbInputs != 1) {
    std::cerr << "This sample currently supports only single-input models. "
              << "Model has " << nbInputs << " inputs.\n";
    return false;
  }

  ITensor *inputTensor = network->getInput(0);
  inputInfo.name = inputTensor->getName();
  inputInfo.dims = inputTensor->getDimensions();

  if (inputInfo.dims.nbDims < 1) {
    std::cerr << "Input tensor has no dimensions?!\n";
    return false;
  }

  if (inputInfo.dims.d[0] < 0) {
    inputInfo.dynamicBatch = true;
  } else {
    inputInfo.dynamicBatch = false;
  }

  // For simplicity, require all other dims to be static.
  for (int i = 1; i < inputInfo.dims.nbDims; ++i) {
    if (inputInfo.dims.d[i] < 0) {
      std::cerr << "Only batch dimension (dim 0) may be dynamic; dim " << i
                << " is also dynamic. This sample doesn't handle that.\n";
      return false;
    }
  }

  std::cout << "Input name: " << inputInfo.name << "\nDims: [";
  for (int i = 0; i < inputInfo.dims.nbDims; ++i) {
    std::cout << inputInfo.dims.d[i]
              << (i + 1 == inputInfo.dims.nbDims ? "" : ", ");
  }
  std::cout << "]  (batch is " << (inputInfo.dynamicBatch ? "dynamic" : "fixed")
            << ")\n";

  return true;
}

// Build engine for a specific batch size using an optimization profile
std::unique_ptr<ICudaEngine> buildEngineForBatch(IRuntime &runtime,
                                                 IBuilder &builder,
                                                 INetworkDefinition &network,
                                                 const InputInfo &inputInfo,
                                                 int batchSize) {
  if (!inputInfo.dynamicBatch) {
    // For a fixed-batch model, building per batch size doesn't make sense.
    // We expect batchSize == fixed batch.
    const int fixedBatch = static_cast<int>(inputInfo.dims.d[0]);
    if (batchSize != fixedBatch) {
      std::cerr << "Requested batch " << batchSize
                << " but model batch is fixed at " << fixedBatch << "\n";
      return nullptr;
    }
  }

  std::unique_ptr<IBuilderConfig> config{builder.createBuilderConfig()};
  if (!config) {
    std::cerr << "Failed to create builder config\n";
    return nullptr;
  }

  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1 GB

  if (builder.platformHasFastFp16()) {
    config->setFlag(BuilderFlag::kFP16);
  }

  if (inputInfo.dynamicBatch) {
    IOptimizationProfile *profile = builder.createOptimizationProfile();
    if (!profile) {
      std::cerr << "Failed to create optimization profile\n";
      return nullptr;
    }

    // Build a profile where min = opt = max = desired batch size.
    Dims profileDims = inputInfo.dims;
    profileDims.d[0] = batchSize;

    if (!profile->setDimensions(inputInfo.name.c_str(),
                                OptProfileSelector::kMIN, profileDims) ||
        !profile->setDimensions(inputInfo.name.c_str(),
                                OptProfileSelector::kOPT, profileDims) ||
        !profile->setDimensions(inputInfo.name.c_str(),
                                OptProfileSelector::kMAX, profileDims)) {
      std::cerr << "Failed to set dimensions on optimization profile\n";
      return nullptr;
    }

    int profileIndex = config->addOptimizationProfile(profile);
    if (profileIndex < 0) {
      std::cerr << "Failed to add optimization profile\n";
      return nullptr;
    }
  }

  std::unique_ptr<IHostMemory> serialized{
      builder.buildSerializedNetwork(network, *config)};
  if (!serialized) {
    std::cerr << "Failed to build serialized network for batch " << batchSize
              << "\n";
    return nullptr;
  }

  std::unique_ptr<ICudaEngine> engine{
      runtime.deserializeCudaEngine(serialized->data(), serialized->size())};
  if (!engine) {
    std::cerr << "Failed to deserialize engine (batch " << batchSize << ")\n";
    return nullptr;
  }

  return engine;
}

struct Buffers {
  std::vector<void *> devicePtrs;
  std::vector<void *> hostPtrs;   // pinned host buffers
  std::vector<std::size_t> bytes; // size per tensor
  std::vector<std::string> tensorNames;
  std::vector<bool> isInput; // input/output tag

  Buffers() = default;
  ~Buffers() { release(); }
  Buffers(const Buffers &other) = delete;
  Buffers &operator=(const Buffers &other) = delete;

  Buffers(Buffers &&other) noexcept { steal(std::move(other)); }
  Buffers &operator=(Buffers &&other) noexcept {
    if (this != &other) {
      release();
      steal(std::move(other));
    }
    return *this;
  }

  void release() {
    for (void *p : devicePtrs) {
      if (p) {
        cudaFree(p);
      }
    }

    for (void *p : hostPtrs) {
      if (p) {
        cudaFreeHost(p);
      }
    }
  }

  void steal(Buffers &&other) {
    devicePtrs = other.devicePtrs;
    hostPtrs = other.hostPtrs;
    bytes = other.bytes;
    tensorNames = other.tensorNames;
    isInput = other.isInput;

    for (auto &p : other.devicePtrs) {
      p = nullptr;
    }

    for (auto &p : other.hostPtrs) {
      p = nullptr;
    }
  }
};

// Allocate device buffers for all IO tensors, using the current context shapes
Buffers allocateBuffers(ICudaEngine &engine, IExecutionContext &context) {
  Buffers buffers;
  const int nbTensors = engine.getNbIOTensors();

  buffers.devicePtrs.reserve(nbTensors);
  buffers.hostPtrs.reserve(nbTensors);
  buffers.bytes.reserve(nbTensors);
  buffers.tensorNames.reserve(nbTensors);
  buffers.isInput.reserve(nbTensors);

  for (int i = 0; i < nbTensors; ++i) {
    const char *name = engine.getIOTensorName(i);
    buffers.tensorNames.emplace_back(name);

    // Shape is resolved in the context (after setInputShape)
    Dims shape = context.getTensorShape(name);
    if (shape.nbDims <= 0) {
      throw std::runtime_error(std::string("Tensor ") + name +
                               " has invalid shape");
    }

    DataType dt = engine.getTensorDataType(name);
    int64_t numElems = volume(shape);
    std::size_t bytes = static_cast<std::size_t>(numElems) * elementSize(dt);

    // Determine IO direction
    const bool input = (engine.getTensorIOMode(name) == TensorIOMode::kINPUT);

    void *dptr = nullptr;
    if (cudaMalloc(&dptr, bytes) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed for tensor " +
                               std::string(name));
    }

    void *hptr = nullptr;
    if (cudaMallocHost(&hptr, bytes) != cudaSuccess) {
      cudaFree(dptr);
      throw std::runtime_error("cudaMallocHost failed for tensor " +
                               std::string(name));
    }

    // Optional: initialize input host buffers so H2D isn't copying
    // uninitialized memory
    if (input) {
      std::memset(hptr, 0, bytes);
    }

    buffers.devicePtrs.push_back(dptr);
    buffers.hostPtrs.push_back(hptr);
    buffers.bytes.push_back(bytes);
    buffers.isInput.push_back(input);
  }

  return buffers;
}

struct Runner {
  std::unique_ptr<IExecutionContext> context{};
  Buffers buffers{};
  cudaStream_t stream{};

  Runner() = default;

  Runner(ICudaEngine &engine, const InputInfo &inputInfo, int batchSize)
      : context{engine.createExecutionContext()} {
    if (!context) {
      throw std::runtime_error("Failed to create execution context");
    }

    if (inputInfo.dynamicBatch) {
      Dims runtimeDims = inputInfo.dims;
      runtimeDims.d[0] = batchSize;

      if (!context->setInputShape(inputInfo.name.c_str(), runtimeDims)) {
        throw std::runtime_error("setInputShape failed");
      }
    }

    buffers = allocateBuffers(engine, *context);

    const int nbTensors = static_cast<int>(buffers.tensorNames.size());
    for (int i = 0; i < nbTensors; ++i) {
      const char *name = buffers.tensorNames[i].c_str();
      if (!context->setTensorAddress(name, buffers.devicePtrs[i])) {
        throw std::runtime_error(std::string("setTensorAddress failed for ") +
                                 name);
      }
    }

    cudaStreamCreate(&stream);
  }

  ~Runner() { cudaStreamDestroy(stream); }
};

inline bool
time_is_up(const std::chrono::time_point<std::chrono::high_resolution_clock> &t,
           const double duration) {
  return std::chrono::duration<double>(
             std::chrono::high_resolution_clock::now() - t)
             .count() > duration;
}

double benchmark_for(Runner &runner, double duration) {
  int iters = 0;
  auto t0 = std::chrono::high_resolution_clock::now();

  while (!time_is_up(t0, duration)) {
    // H2D for inputs
    for (int i = 0; i < (int)runner.buffers.tensorNames.size(); ++i) {
      if (!runner.buffers.isInput[i]) {
        continue;
      }
      cudaMemcpyAsync(runner.buffers.devicePtrs[i], runner.buffers.hostPtrs[i],
                      runner.buffers.bytes[i], cudaMemcpyHostToDevice,
                      runner.stream);
    }

    if (!runner.context->enqueueV3(runner.stream)) {
      throw std::runtime_error("enqueueV3 failed during benchmark");
    }

    // D2H for outputs
    for (int i = 0; i < (int)runner.buffers.tensorNames.size(); ++i) {
      if (runner.buffers.isInput[i]) {
        continue;
      }
      cudaMemcpyAsync(runner.buffers.hostPtrs[i], runner.buffers.devicePtrs[i],
                      runner.buffers.bytes[i], cudaMemcpyDeviceToHost,
                      runner.stream);
    }

    iters++;
  }

  cudaStreamSynchronize(runner.stream);
  auto t1 = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// Benchmark an engine at a particular batch size (ms / image)
double benchmarkEngine(ICudaEngine &engine, const InputInfo &inputInfo,
                       int batchSize) {
  std::vector<Runner> runners;
  runners.emplace_back(engine, inputInfo, batchSize);

  // Warmup.
  benchmark_for(runners[0], 2);

  // Timed iterations.
  const double msPerIter = benchmark_for(runners[0], 20);

  return msPerIter / static_cast<double>(batchSize);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " model.onnx [batch1 batch2 ...]\n";
    return 1;
  }

  std::string onnxPath = argv[1];

  std::vector<int> candidateBatches;
  for (int i = 2; i < argc; ++i) {
    const int batch = std::stoi(argv[i]);
    if (batch <= 0) {
      std::cerr << "Batch must be > 0\n";
      return EXIT_FAILURE;
    }
    candidateBatches.push_back(batch);
  }

  initLibNvInferPlugins(&gLogger, "");

  std::unique_ptr<IBuilder> builder;
  std::unique_ptr<INetworkDefinition> network;
  std::unique_ptr<nvonnxparser::IParser> parser;
  InputInfo inputInfo;

  if (!parseOnnx(onnxPath, builder, network, parser, inputInfo)) {
    return EXIT_FAILURE;
  }

  if (!inputInfo.dynamicBatch) {
    candidateBatches.clear();
    candidateBatches.push_back(inputInfo.dims.d[0]);
  }

  std::unique_ptr<IRuntime> runtime{createInferRuntime(gLogger)};
  if (!runtime) {
    std::cerr << "Failed to create runtime\n";
    return EXIT_FAILURE;
  }

  double bestMsPerImage = 1e30;
  int bestBatch = -1;

  for (int b : candidateBatches) {
    try {
      std::cout << "\n=== Building engine for batch " << b << " ===\n";
      std::unique_ptr<ICudaEngine> engine =
          buildEngineForBatch(*runtime, *builder, *network, inputInfo, b);
      if (!engine) {
        std::cerr << "Skipping batch " << b << " (build failed)\n";
        continue;
      }

      std::cout << "Benchmarking batch " << b << "...\n";
      double msPerImage = benchmarkEngine(*engine, inputInfo, b);
      double fps = 1000.0 / msPerImage;

      std::cout << "Batch " << b << ": " << msPerImage << " ms / image  ("
                << fps << " images/s)\n";

      if (msPerImage < bestMsPerImage) {
        bestMsPerImage = msPerImage;
        bestBatch = b;
      }
    } catch (const std::exception &e) {
      std::cerr << "Error at batch " << b << ": " << e.what() << "\n";
    }
  }

  if (bestBatch < 0) {
    std::cerr << "Failed to benchmark any batch size\n";
    return EXIT_FAILURE;
  }

  std::cout << "\n=== Best batch size: " << bestBatch << " (" << bestMsPerImage
            << " ms / image) ===\n";

  return EXIT_SUCCESS;
}
