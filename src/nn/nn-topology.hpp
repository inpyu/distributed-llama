#ifndef NN_TOPOLOGY_H
#define NN_TOPOLOGY_H

#include "nn-core.hpp"
#include <stdexcept>

typedef struct {
    NnUint globalRank;
    NnUint ppRank;
    NnUint tpRank;
    NnUint tpGroupStart;
    NnUint tpGroupEnd;
} NnNodePlacement;

class NnParallelTopology {
public:
    NnUint nNodes;
    NnUint ppSize;
    NnUint tpSize;

    NnParallelTopology(NnUint nNodes, NnUint ppSize)
        : nNodes(nNodes), ppSize(ppSize), tpSize(0) {
        if (nNodes < 1) {
            throw std::runtime_error("nNodes must be >= 1");
        }
        if (ppSize < 1) {
            throw std::runtime_error("ppSize must be >= 1");
        }
        if (nNodes % ppSize != 0) {
            throw std::runtime_error("Invalid topology: nNodes must be divisible by ppSize");
        }
        this->tpSize = nNodes / ppSize;
    }

    NnNodePlacement getPlacement(NnUint globalRank) const {
        if (globalRank >= nNodes) {
            throw std::runtime_error("global rank out of range");
        }
        const NnUint ppRank = globalRank / tpSize;
        const NnUint tpRank = globalRank % tpSize;
        const NnUint tpGroupStart = ppRank * tpSize;
        const NnUint tpGroupEnd = tpGroupStart + tpSize;
        return NnNodePlacement{globalRank, ppRank, tpRank, tpGroupStart, tpGroupEnd};
    }

    NnUint getGlobalRank(NnUint ppRank, NnUint tpRank) const {
        if (ppRank >= ppSize) {
            throw std::runtime_error("pp rank out of range");
        }
        if (tpRank >= tpSize) {
            throw std::runtime_error("tp rank out of range");
        }
        return ppRank * tpSize + tpRank;
    }
};

static inline NnParallelTopology createPPxTPTopology(NnUint nNodes, NnUint ppSize) {
    return NnParallelTopology(nNodes, ppSize);
}

#endif
