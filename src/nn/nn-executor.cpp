#include <cassert>
#include <cstring>
#include "nn-executor.hpp"

static inline void executeStep(NnExecutorStep *step, NnUint nThreads, NnExecutorThread *thread, NnExecutorContext *context);

void NnFakeNodeSynchronizer::sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) {
    // Nothing
}

NnNetExecution::NnNetExecution(NnUint nThreads, NnNetConfig *netConfig) {
    this->nThreads = nThreads;
    this->nBatches = netConfig->nBatches;
    this->nPipes = netConfig->nPipes;
    this->batchSize = 0; // This value must be overwritten before calling forward

    pipes = new NnByte *[netConfig->nPipes];
    for (NnUint pipeIndex = 0; pipeIndex < netConfig->nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &netConfig->pipes[pipeIndex];
        NnByte *pipe = new NnByte[pipeConfig->size.nBytes];
        std::memset(pipe, 0, pipeConfig->size.nBytes);
        pipes[pipeIndex] = pipe;
    }
}

NnNetExecution::~NnNetExecution() {
    for (NnUint pipeIndex = 0; pipeIndex < nPipes; pipeIndex++)
        delete[] pipes[pipeIndex];
    delete[] pipes;
}

void NnNetExecution::setBatchSize(NnUint batchSize) {
    assert(batchSize <= nBatches);
    this->batchSize = batchSize;
}

NnExecutorDevice::NnExecutorDevice(NnDevice *device, int segmentFrom, int segmentTo) {
    this->device = std::unique_ptr<NnDevice>(device);
    this->segmentFrom = segmentFrom;
    this->segmentTo = segmentTo;
}

NnExecutorException::NnExecutorException(const std::string message)
    : std::runtime_error(message) 
{}

NnExecutor::NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, std::vector<NnExecutorDevice> *devices, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer, bool benchmark)
    : segments(nodeConfig->nSegments), steps()
{
    NnUint maxNThreads = 0;
    for (NnExecutorDevice &d : *devices) {
        if (d.device->maxNThreads() > maxNThreads)
            maxNThreads = d.device->maxNThreads();
    }
    if (netExecution->nThreads > maxNThreads)
        throw std::invalid_argument("This configuration supports max " + std::to_string(maxNThreads) + " threads");

    this->netExecution = netExecution;
    this->nodeConfig = nodeConfig;

    bool useSynchronizer = netConfig->nNodes > 1;
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnDevice *device = nullptr;
        for (NnExecutorDevice &d : *devices) {
            if (
                (d.segmentFrom == -1 && d.segmentTo == -1) ||
                (segmentIndex >= d.segmentFrom && segmentIndex <= d.segmentTo)
            ) {
                device = d.device.get();
                break;
            }
        }
        if (device == nullptr)
            throw std::invalid_argument("Cannot locate device for segment " + std::to_string(segmentIndex));

        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        if (segmentConfig->nOps > 0) {
            NnDeviceSegment *segment = device->createSegment(segmentIndex);
            segments[segmentIndex] = std::unique_ptr<NnDeviceSegment>(segment);

            for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++)
                steps.push_back(NnExecutorStep{ STEP_EXECUTE_OP, segment, opIndex, &segmentConfig->ops[opIndex] });
        }
        if (useSynchronizer && segmentConfig->nSyncs > 0)
            steps.push_back(NnExecutorStep{ STEP_SYNC_NODES, nullptr, segmentIndex, nullptr });
    }

    steps.shrink_to_fit();

    context.nThreads = netExecution->nThreads;
    context.synchronizer = synchronizer;
    context.nSteps = (NnUint)steps.size();
    context.steps = steps.data();
    context.epoch.store(0);
    context.doneRunThreadCount.store(0);
    if (benchmark)
        context.timer = new Timer();
    else
        context.timer = nullptr;

    context.isAlive.store(true);
    context.isShutdown.store(false);
    context.isRunDone.store(true);

    threads.resize(netExecution->nThreads);
    for (NnUint threadIndex = 0; threadIndex < netExecution->nThreads; threadIndex++) {
        threads[threadIndex].threadIndex = threadIndex;
        threads[threadIndex].context = &context;
    }

    threadHandles.reserve(netExecution->nThreads);
    for (NnUint threadIndex = 0; threadIndex < netExecution->nThreads; threadIndex++) {
        threadHandles.emplace_back([this, threadIndex]() {
            NnExecutorThread *thread = &this->threads[threadIndex];
            NnExecutorContext *ctx = thread->context;
            NnUint localEpoch = 0;

            while (true) {
                {
                    std::unique_lock<std::mutex> lock(ctx->mutex);
                    ctx->cv.wait(lock, [ctx, localEpoch]() {
                        return ctx->isShutdown.load() || ctx->epoch.load() != localEpoch;
                    });
                }

                if (ctx->isShutdown.load())
                    break;

                localEpoch = ctx->epoch.load();

                NnUint nThreads = ctx->nThreads;
                NnUint doneCount = nThreads - 1;

                while (ctx->isAlive.load()) {
                    const NnUint currentStepIndex = ctx->currentStepIndex.load();
                    if (currentStepIndex == ctx->nSteps)
                        break;

                    NnExecutorStep *step = &ctx->steps[currentStepIndex];
                    try {
                        executeStep(step, nThreads, thread, ctx);
                    } catch (const std::runtime_error &e) {
                        ctx->isAlive.store(false);
                        printf("🚨 Execution error: %s\n", e.what());
                        ctx->cv.notify_all();
                        break;
                    }

                    NnUint currentCount = ctx->doneThreadCount.fetch_add(1);
                    if (currentCount == doneCount) {
                        if (ctx->timer != nullptr) {
                            NnUint time = ctx->timer->elapsedMicroseconds();
                            ctx->totalTime[step->type] += time;
                            ctx->timer->reset();
                        }

                        ctx->doneThreadCount.store(0);
                        ctx->currentStepIndex.fetch_add(1);
                        ctx->cv.notify_all();
                    } else {
                        std::unique_lock<std::mutex> lock(ctx->mutex);
                        ctx->cv.wait(lock, [ctx, currentStepIndex, localEpoch]() {
                            return ctx->isShutdown.load() ||
                                   !ctx->isAlive.load() ||
                                   ctx->epoch.load() != localEpoch ||
                                   ctx->currentStepIndex.load() != currentStepIndex;
                        });
                        if (ctx->isShutdown.load() || ctx->epoch.load() != localEpoch)
                            break;
                    }
                }

                NnUint runCount = ctx->doneRunThreadCount.fetch_add(1);
                if (runCount == doneCount) {
                    ctx->isRunDone.store(true);
                    ctx->cv.notify_all();
                }
            }
        });
    }
}

NnExecutor::~NnExecutor() {
    if (context.timer != nullptr)
        delete context.timer;

    context.isShutdown.store(true);
    context.epoch.fetch_add(1);
    context.cv.notify_all();
    for (std::thread &t : threadHandles) {
        if (t.joinable())
            t.join();
    }
}

void NnExecutor::loadWeight(const char *name, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        for (NnUint i = 0; i < segmentConfig->nOps; i++) {
            NnOpConfig *opConfig = &segmentConfig->ops[i];
            if (opConfig->index == opIndex && std::strcmp(opConfig->name, name) == 0) {
                NnDeviceSegment *segment = segments[segmentIndex].get();
                assert(segment != nullptr);
                segment->loadWeight(i, offset, nBytes, weight);
                return;
            }
        }
    }
    throw std::invalid_argument("Cannot locate op by name: " + std::string(name));
}

static inline void executeStep(NnExecutorStep *step, NnUint nThreads, NnExecutorThread *thread, NnExecutorContext *context) {
    if (step->type == STEP_EXECUTE_OP) {
        step->segment->forward(step->arg0, nThreads, thread->threadIndex, context->batchSize);
    } else if (step->type == STEP_SYNC_NODES) {
        context->synchronizer->sync(step->arg0, nThreads, thread->threadIndex);
    } else {
        throw std::invalid_argument("Unsupported step type");
    }
}

void NnExecutor::forward() {
    assert(netExecution->batchSize > 0);

    {
        std::lock_guard<std::mutex> lock(context.mutex);
        context.isAlive.store(true);
        context.currentStepIndex.store(0);
        context.doneThreadCount.store(0);
        context.doneRunThreadCount.store(0);
        context.isRunDone.store(false);
        context.batchSize = netExecution->batchSize;

        if (context.timer != nullptr) {
            std::memset(context.totalTime, 0, sizeof(context.totalTime));
            context.timer->reset();
        }

        context.epoch.fetch_add(1);
    }
    context.cv.notify_all();

    std::unique_lock<std::mutex> lock(context.mutex);
    context.cv.wait(lock, [this]() {
        return this->context.isShutdown.load() ||
               this->context.isRunDone.load() ||
               !this->context.isAlive.load();
    });

    if (!context.isAlive.load())
        throw NnExecutorException("Execution failed in one of the threads");
}

NnUint NnExecutor::getTotalTime(NnExecutorStepType type) {
    assert((NnUint)type < N_STEP_TYPES);
    return context.totalTime[type];
}
