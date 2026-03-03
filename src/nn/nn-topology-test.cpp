#include "nn-topology.hpp"
#include <cstdio>

static void assertTrue(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void testPp1Mapping() {
    NnParallelTopology topology = createPPxTPTopology(8, 1);
    assertTrue(topology.ppSize == 1, "ppSize mismatch for PP=1");
    assertTrue(topology.tpSize == 8, "tpSize mismatch for PP=1");

    for (NnUint rank = 0; rank < 8; rank++) {
        NnNodePlacement p = topology.getPlacement(rank);
        assertTrue(p.ppRank == 0, "ppRank should be zero for PP=1");
        assertTrue(p.tpRank == rank, "tpRank should equal global rank for PP=1");
        assertTrue(p.tpGroupStart == 0, "tpGroupStart should be 0 for PP=1");
        assertTrue(p.tpGroupEnd == 8, "tpGroupEnd should be nNodes for PP=1");
    }
}

static void testPp2Mapping() {
    NnParallelTopology topology = createPPxTPTopology(8, 2);
    assertTrue(topology.ppSize == 2, "ppSize mismatch for PP=2");
    assertTrue(topology.tpSize == 4, "tpSize mismatch for PP=2");

    NnNodePlacement p0 = topology.getPlacement(0);
    assertTrue(p0.ppRank == 0 && p0.tpRank == 0, "rank0 placement mismatch");
    assertTrue(p0.tpGroupStart == 0 && p0.tpGroupEnd == 4, "rank0 tp group mismatch");

    NnNodePlacement p5 = topology.getPlacement(5);
    assertTrue(p5.ppRank == 1 && p5.tpRank == 1, "rank5 placement mismatch");
    assertTrue(p5.tpGroupStart == 4 && p5.tpGroupEnd == 8, "rank5 tp group mismatch");
}

static void testRoundTrip() {
    NnParallelTopology topology = createPPxTPTopology(16, 4);
    for (NnUint pp = 0; pp < topology.ppSize; pp++) {
        for (NnUint tp = 0; tp < topology.tpSize; tp++) {
            NnUint rank = topology.getGlobalRank(pp, tp);
            NnNodePlacement p = topology.getPlacement(rank);
            assertTrue(p.ppRank == pp, "roundtrip pp mismatch");
            assertTrue(p.tpRank == tp, "roundtrip tp mismatch");
        }
    }
}

static void testInvalidInputs() {
    bool thrown = false;
    try {
        createPPxTPTopology(8, 3);
    } catch (const std::runtime_error &) {
        thrown = true;
    }
    assertTrue(thrown, "expected divisibility validation failure");

    thrown = false;
    try {
        createPPxTPTopology(8, 0);
    } catch (const std::runtime_error &) {
        thrown = true;
    }
    assertTrue(thrown, "expected ppSize validation failure");

    thrown = false;
    try {
        NnParallelTopology topology = createPPxTPTopology(8, 2);
        topology.getPlacement(8);
    } catch (const std::runtime_error &) {
        thrown = true;
    }
    assertTrue(thrown, "expected rank bounds validation failure");
}

int main() {
    testPp1Mapping();
    printf("✅          testPp1Mapping passed\n");

    testPp2Mapping();
    printf("✅          testPp2Mapping passed\n");

    testRoundTrip();
    printf("✅           testRoundTrip passed\n");

    testInvalidInputs();
    printf("✅       testInvalidInputs passed\n");
    return 0;
}
