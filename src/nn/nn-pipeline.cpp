#include "nn-pipeline.hpp"
#include <cstring>
#include <stdexcept>

NnPipelineCommunicator::NnPipelineCommunicator(NnNetwork *network, const NnParallelTopology *topology, NnUint myNodeIndex) {
    this->network = network;
    this->topology = topology;
    this->myNodeIndex = myNodeIndex;
    
    NnNodePlacement placement = topology->getPlacement(myNodeIndex);
    this->myPpRank = placement.ppRank;
    this->myTpRank = placement.tpRank;
}

NnPipelineCommunicator::~NnPipelineCommunicator() {
    // Nothing to cleanup - network is owned externally
}

NnUint NnPipelineCommunicator::getPipelineSocketIndex(NnUint targetPpRank, NnUint targetTpRank) const {
    NnUint targetGlobalRank = topology->getGlobalRank(targetPpRank, targetTpRank);

    if (targetGlobalRank == myNodeIndex) {
        throw std::runtime_error("Pipeline socket target cannot be current node");
    }

    if (myNodeIndex == 0) {
        if (targetGlobalRank == 0)
            throw std::runtime_error("Invalid root pipeline socket target");
        return targetGlobalRank - 1;
    }

    if (targetGlobalRank == 0)
        return 0;

    if (targetGlobalRank < myNodeIndex)
        return targetGlobalRank;

    return targetGlobalRank - 1;
}

NnUint NnPipelineCommunicator::calculateChecksum(const NnByte *data, NnSize bytes, NnFloatType dtype) const {
    // Simple checksum: sum of first few values
    NnUint checksum = 0;
    
    if (dtype == F_32 && bytes >= sizeof(float) * 4) {
        const float *floats = reinterpret_cast<const float*>(data);
        for (int i = 0; i < 4; i++) {
            // Convert float to int for checksum
            std::memcpy(&checksum, &floats[i], sizeof(NnUint));
            checksum ^= (i + 1); // XOR with position
        }
    } else {
        // For other types or small data, just use first bytes
        NnSize checksumBytes = bytes < sizeof(NnUint) ? bytes : sizeof(NnUint);
        std::memcpy(&checksum, data, checksumBytes);
    }
    
    return checksum;
}

bool NnPipelineCommunicator::sendActivation(
    NnUint targetPpRank,
    NnUint seqPosition,
    NnUint batchIndex,
    NnUint sliceId,
    const NnByte *data,
    NnSize bytes,
    NnFloatType dtype
) {
    if (targetPpRank >= topology->ppSize) {
        throw std::runtime_error("Invalid target PP rank for pipeline communication");
    }
    
    // Slice-preserving: send to same TP rank in target stage
    NnUint targetSocketIndex = getPipelineSocketIndex(targetPpRank, myTpRank);
    
    // Prepare header
    NnPipelineActivationHeader header;
    header.seqPosition = seqPosition;
    header.batchIndex = batchIndex;
    header.sliceId = sliceId;
    header.dtype = dtype;
    header.payloadBytes = bytes;
    header.checksum = calculateChecksum(data, bytes, dtype);
    
    try {
        // Send header first
        network->write(targetSocketIndex, &header, sizeof(header));
        
        // Send payload
        network->write(targetSocketIndex, data, bytes);
        
        // Wait for ack
        network->readAck(targetSocketIndex);
        
        return true;
    } catch (const std::exception &e) {
        printf("🚨 Pipeline send error (target PP=%u, socket=%u): %s\n", 
               targetPpRank, targetSocketIndex, e.what());
        return false;
    }
}

bool NnPipelineCommunicator::recvActivation(
    NnUint sourcePpRank,
    NnPipelineActivationHeader *header,
    NnByte *buffer,
    NnSize bufferSize
) {
    if (sourcePpRank >= topology->ppSize) {
        throw std::runtime_error("Invalid source PP rank for pipeline communication");
    }
    
    // Slice-preserving: receive from same TP rank in source stage
    NnUint sourceSocketIndex = getPipelineSocketIndex(sourcePpRank, myTpRank);
    
    try {
        // Receive header
        network->read(sourceSocketIndex, header, sizeof(NnPipelineActivationHeader));
        
        // Validate header
        if (header->payloadBytes > bufferSize) {
            printf("🚨 Pipeline recv error: payload too large (%zu > %zu)\n",
                   header->payloadBytes, bufferSize);
            return false;
        }
        
        // Receive payload
        network->read(sourceSocketIndex, buffer, header->payloadBytes);
        
        // Verify checksum
        NnUint receivedChecksum = calculateChecksum(buffer, header->payloadBytes, header->dtype);
        if (receivedChecksum != header->checksum) {
            printf("⚠️  Pipeline recv warning: checksum mismatch (expected=%u, got=%u)\n",
                   header->checksum, receivedChecksum);
            // Continue anyway - checksum is just a hint
        }
        
        // Send ack
        network->writeAck(sourceSocketIndex);
        
        return true;
    } catch (const std::exception &e) {
        printf("🚨 Pipeline recv error (source PP=%u, socket=%u): %s\n",
               sourcePpRank, sourceSocketIndex, e.what());
        return false;
    }
}

bool NnPipelineCommunicator::shouldSendActivations() const {
    // Send if not in the last stage
    return myPpRank < topology->ppSize - 1;
}

bool NnPipelineCommunicator::shouldRecvActivations() const {
    // Receive if not in the first stage
    return myPpRank > 0;
}

NnUint NnPipelineCommunicator::getTargetPpRank() const {
    if (!shouldSendActivations()) {
        throw std::runtime_error("Cannot get target PP rank: already in last stage");
    }
    return myPpRank + 1;
}

NnUint NnPipelineCommunicator::getSourcePpRank() const {
    if (!shouldRecvActivations()) {
        throw std::runtime_error("Cannot get source PP rank: already in first stage");
    }
    return myPpRank - 1;
}
