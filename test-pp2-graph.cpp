#include "src/llm.hpp"
#include "src/nn/nn-topology.hpp"
#include <cstdio>
#include <cstring>

// Simple test to verify PP=2 graph construction works
int main() {
    printf("🧪 Testing PP=2 graph construction...\n\n");
    
    // Create minimal LlmHeader for testing
    LlmHeader header;
    std::memset(&header, 0, sizeof(LlmHeader));
    
    header.archType = LLAMA;
    header.ropeType = ROPE_LLAMA;
    header.hiddenAct = HIDDEN_ACT_SILU;
    header.dim = 512;
    header.hiddenDim = 1024;
    header.nLayers = 8;  // 8 layers so PP=2 splits into 4 layers each
    header.nHeads = 8;
    header.nKvHeads = 8;
    header.headDim = 64;
    header.vocabSize = 1000;
    header.seqLen = 128;
    header.qDim = 512;
    header.kvDim = 512;
    header.nExperts = 0;
    header.nActiveExperts = 0;
    header.ropeTheta = 10000.0f;
    header.normEpsilon = 1e-5f;
    header.weightType = F_32;
    header.syncType = F_32;
    
    // Test 1: PP=1 (backward compatibility)
    printf("Test 1: PP=1 (backward compatibility)\n");
    {
        NnParallelTopology topology = createPPxTPTopology(2, 1);  // 2 nodes, PP=1
        LlmNet net = buildLlmNet(&header, topology, 1);
        
        // Verify all nodes process all layers
        for (NnUint i = 0; i < 2; i++) {
            NnNodePlacement p = topology.getPlacement(i);
            printf("  Node %u: ppRank=%u, tpRank=%u, tpGroup=[%u,%u)\n",
                   i, p.ppRank, p.tpRank, p.tpGroupStart, p.tpGroupEnd);
            
            if (p.ppRank != 0) {
                printf("❌ PP=1 should have ppRank=0 for all nodes\n");
                return 1;
            }
        }
        
        releaseLlmNet(&net);
        printf("✅ PP=1 graph construction successful\n\n");
    }
    
    // Test 2: PP=2 with 2 nodes
    printf("Test 2: PP=2 with 2 nodes (1 node per stage)\n");
    {
        NnParallelTopology topology = createPPxTPTopology(2, 2);  // 2 nodes, PP=2
        LlmNet net = buildLlmNet(&header, topology, 1);
        
        // Verify nodes are split into stages
        for (NnUint i = 0; i < 2; i++) {
            NnNodePlacement p = topology.getPlacement(i);
            printf("  Node %u: ppRank=%u, tpRank=%u, tpGroup=[%u,%u)\n",
                   i, p.ppRank, p.tpRank, p.tpGroupStart, p.tpGroupEnd);
            
            NnUint expectedPpRank = i;  // Node 0 -> stage 0, Node 1 -> stage 1
            if (p.ppRank != expectedPpRank) {
                printf("❌ Node %u should have ppRank=%u, got %u\n", i, expectedPpRank, p.ppRank);
                return 1;
            }
        }
        
        releaseLlmNet(&net);
        printf("✅ PP=2 graph construction successful\n\n");
    }
    
    // Test 3: PP=2 with 4 nodes (TP=2 per stage)
    printf("Test 3: PP=2 x TP=2 (4 nodes total)\n");
    {
        NnParallelTopology topology = createPPxTPTopology(4, 2);  // 4 nodes, PP=2
        LlmNet net = buildLlmNet(&header, topology, 1);
        
        // Verify topology
        for (NnUint i = 0; i < 4; i++) {
            NnNodePlacement p = topology.getPlacement(i);
            printf("  Node %u: ppRank=%u, tpRank=%u, tpGroup=[%u,%u)\n",
                   i, p.ppRank, p.tpRank, p.tpGroupStart, p.tpGroupEnd);
            
            NnUint expectedPpRank = i / 2;  // Nodes 0-1 -> stage 0, Nodes 2-3 -> stage 1
            NnUint expectedTpRank = i % 2;
            
            if (p.ppRank != expectedPpRank) {
                printf("❌ Node %u should have ppRank=%u, got %u\n", i, expectedPpRank, p.ppRank);
                return 1;
            }
            if (p.tpRank != expectedTpRank) {
                printf("❌ Node %u should have tpRank=%u, got %u\n", i, expectedTpRank, p.tpRank);
                return 1;
            }
        }
        
        releaseLlmNet(&net);
        printf("✅ PP=2 x TP=2 graph construction successful\n\n");
    }
    
    // Test 4: PP=2 with 8 nodes (TP=4 per stage) - target topology from plan.md
    printf("Test 4: PP=2 x TP=4 (8 nodes total - target topology)\n");
    {
        NnParallelTopology topology = createPPxTPTopology(8, 2);  // 8 nodes, PP=2
        LlmNet net = buildLlmNet(&header, topology, 1);
        
        // Verify topology matches plan.md: Stage 0 = nodes 0-3, Stage 1 = nodes 4-7
        for (NnUint i = 0; i < 8; i++) {
            NnNodePlacement p = topology.getPlacement(i);
            printf("  Node %u: ppRank=%u, tpRank=%u, tpGroup=[%u,%u)\n",
                   i, p.ppRank, p.tpRank, p.tpGroupStart, p.tpGroupEnd);
            
            NnUint expectedPpRank = i / 4;  // Nodes 0-3 -> stage 0, Nodes 4-7 -> stage 1
            NnUint expectedTpRank = i % 4;
            
            if (p.ppRank != expectedPpRank) {
                printf("❌ Node %u should have ppRank=%u, got %u\n", i, expectedPpRank, p.ppRank);
                return 1;
            }
            if (p.tpRank != expectedTpRank) {
                printf("❌ Node %u should have tpRank=%u, got %u\n", i, expectedTpRank, p.tpRank);
                return 1;
            }
        }
        
        releaseLlmNet(&net);
        printf("✅ PP=2 x TP=4 graph construction successful\n\n");
    }
    
    printf("🎉 All PP=2 graph construction tests passed!\n");
    printf("\n");
    printf("⚠️  Note: Graph construction works, but runtime pipeline communication\n");
    printf("   is not yet implemented. Full PP=2 execution requires M4.5 integration.\n");
    
    return 0;
}
