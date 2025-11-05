#include "nn-network.hpp"
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    printf("🔍 Network Bottleneck Analysis Tool\n");
    printf("====================================\n");
    
    // This is a simple test program to demonstrate the network monitoring capabilities
    // In a real scenario, you would use this with your distributed inference setup
    
    printf("\n📊 To enable network performance monitoring in your distributed inference:\n");
    printf("1. Enable monitoring: network->enablePerformanceMonitoring(true);\n");
    printf("2. Run your inference workload\n");
    printf("3. Print performance report: network->printPerformanceReport();\n");
    printf("4. Print bottleneck analysis: network->printBottleneckAnalysis();\n");
    
    printf("\n🔧 Expected output format:\n");
    printf("📊 === Network Performance Report ===\n");
    printf("Socket | Operations | Total MB | Avg Latency | Max Latency | Min Latency | Bandwidth\n");
    printf("-------|------------|----------|-------------|-------------|-------------|----------\n");
    printf("   0   |     1250   |   45.67   |    2.34 ms  |    15.67 ms  |     0.12 ms  |  156.78 Mbps\n");
    printf("   1   |     1250   |   45.67   |    3.45 ms  |    18.23 ms  |     0.15 ms  |  105.45 Mbps\n");
    
    printf("\n🔍 === Network Bottleneck Analysis ===\n");
    printf("🐌 Slowest Socket: 1 (Avg Latency: 3.45 ms)\n");
    printf("Socket 0: P50=1.23ms, P95=8.45ms, P99=12.34ms\n");
    printf("Socket 1: P50=2.34ms, P95=12.45ms, P99=16.78ms\n");
    printf("⚠️  Socket 1 shows high latency variance (P95 >> P50) - potential network congestion\n");
    
    printf("\n📈 Operation Analysis:\n");
    printf("  SYNC_WITH_ROOT: 500 ops, 22.50 MB, 2.45 ms avg\n");
    printf("  readMany: 750 ops, 23.17 MB, 3.12 ms avg\n");
    
    printf("\n💡 Recommendations based on your 8-node setup:\n");
    printf("- Monitor which specific sockets show high latency variance\n");
    printf("- Check if certain sync operations (SYNC_WITH_ROOT vs SYNC_NODE_SLICES) are slower\n");
    printf("- Identify if bandwidth limitations are causing bottlenecks\n");
    printf("- Consider network topology optimization if specific node pairs are consistently slow\n");
    
    return 0;
}

