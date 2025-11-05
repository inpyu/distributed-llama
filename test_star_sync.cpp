// Test program to validate Star sync logic
#include <stdio.h>
#include <string.h>

void test_star_sync() {
    const int nNodes = 8;
    const int nThreads = 4;
    const int sliceBytes = 100;
    const int nBytes = sliceBytes * nNodes;
    
    // Test thread distribution
    printf("=== Thread Distribution Test ===\n");
    for (int nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
        printf("\nNode %d:\n", nodeIndex);
        
        if (nodeIndex == 0) {
            // ROOT
            int nWorkers = nNodes - 1;
            for (int threadIndex = 0; threadIndex < nThreads; threadIndex++) {
                int workersPerThread = nWorkers / nThreads + (nWorkers % nThreads > threadIndex ? 1 : 0);
                printf("  Thread %d: handles %d workers - ", threadIndex, workersPerThread);
                
                for (int i = 0; i < workersPerThread; i++) {
                    int workerIdx = threadIndex + i * nThreads + 1;
                    if (workerIdx < nNodes) {
                        int socketIndex = workerIdx - 1;
                        printf("Worker %d (socket %d) ", workerIdx, socketIndex);
                    }
                }
                printf("\n");
            }
        } else {
            // WORKER
            printf("  Thread 0: sends to ROOT (socket 0)\n");
            printf("  Thread 1-3: wait\n");
        }
    }
    
    // Test slice positions
    printf("\n=== Slice Position Test ===\n");
    for (int i = 0; i < nNodes; i++) {
        printf("Node %d slice: offset %d bytes\n", i, i * sliceBytes);
    }
}

int main() {
    test_star_sync();
    return 0;
}


