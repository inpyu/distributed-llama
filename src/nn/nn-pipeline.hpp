#ifndef NN_PIPELINE_H
#define NN_PIPELINE_H

#include "nn-core.hpp"
#include "nn-network.hpp"
#include "nn-topology.hpp"

// Pipeline activation transfer packet header
// Used for sending activations between pipeline stages
typedef struct {
    NnUint seqPosition;      // Sequence position of this activation
    NnUint batchIndex;       // Batch index (for batched inference)
    NnUint sliceId;          // TP slice ID (which TP rank this belongs to)
    NnFloatType dtype;       // Data type of the payload
    NnSize payloadBytes;     // Size of activation data in bytes
    NnUint checksum;         // Simple checksum for data integrity (sum of first 4 floats)
} NnPipelineActivationHeader;

// Pipeline communicator for stage-to-stage activation transfer
class NnPipelineCommunicator {
private:
    NnNetwork *network;
    const NnParallelTopology *topology;
    NnUint myNodeIndex;
    NnUint myPpRank;
    NnUint myTpRank;
    
    // Timeout configuration (in milliseconds)
    static const NnUint DEFAULT_SEND_TIMEOUT_MS = 5000;
    static const NnUint DEFAULT_RECV_TIMEOUT_MS = 10000;
    
    // Calculate socket index for pipeline communication
    // Returns the socket index to communicate with the corresponding node in next/prev stage
    NnUint getPipelineSocketIndex(NnUint targetPpRank, NnUint targetTpRank) const;
    
    // Simple checksum for data integrity
    NnUint calculateChecksum(const NnByte *data, NnSize bytes, NnFloatType dtype) const;

public:
    NnPipelineCommunicator(NnNetwork *network, const NnParallelTopology *topology, NnUint myNodeIndex);
    ~NnPipelineCommunicator();
    
    // Send activation to the next pipeline stage
    // - targetPpRank: Pipeline rank of the target stage
    // - seqPosition: Sequence position
    // - batchIndex: Batch index
    // - sliceId: TP slice ID (usually same as myTpRank for slice-preserving)
    // - data: Activation data
    // - bytes: Size of data
    // - dtype: Data type
    // Returns: true on success, false on timeout/error
    bool sendActivation(
        NnUint targetPpRank,
        NnUint seqPosition,
        NnUint batchIndex,
        NnUint sliceId,
        const NnByte *data,
        NnSize bytes,
        NnFloatType dtype
    );
    
    // Receive activation from the previous pipeline stage
    // - sourcePpRank: Pipeline rank of the source stage
    // - header: Output parameter for received header
    // - buffer: Buffer to receive activation data (must be pre-allocated)
    // - bufferSize: Size of the buffer
    // Returns: true on success, false on timeout/error
    bool recvActivation(
        NnUint sourcePpRank,
        NnPipelineActivationHeader *header,
        NnByte *buffer,
        NnSize bufferSize
    );
    
    // Check if this node should send activations (not in last stage)
    bool shouldSendActivations() const;
    
    // Check if this node should receive activations (not in first stage)
    bool shouldRecvActivations() const;
    
    // Get the target PP rank for sending (myPpRank + 1)
    NnUint getTargetPpRank() const;
    
    // Get the source PP rank for receiving (myPpRank - 1)
    NnUint getSourcePpRank() const;
};

#endif
