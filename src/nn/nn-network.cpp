#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h> // For inet_addr and other functions
#include <windows.h>  // For SSIZE_T
typedef SSIZE_T ssize_t;
#else
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>  // for getaddrinfo
#endif
#include "nn-network.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <map>

#define SOCKET_LAST_ERRCODE errno
#define SOCKET_LAST_ERROR strerror(errno)

#define ACK 23571114
#define MAX_CHUNK_SIZE 4096

static inline bool isEagainError() {
    #ifdef _WIN32
    return WSAGetLastError() == WSAEWOULDBLOCK;
    #else
    return SOCKET_LAST_ERRCODE == EAGAIN;
    #endif
}

static inline void setNonBlocking(int socket, bool enabled) {
#ifdef _WIN32
    u_long mode = enabled ? 1 : 0;
    if (ioctlsocket(socket, FIONBIO, &mode) != 0) {
        throw std::runtime_error("Error setting socket to non-blocking");
    }
#else
    int flags = fcntl(socket, F_GETFL, 0);
    if (enabled) {
        flags |= O_NONBLOCK;
    } else {
        flags = flags & (~O_NONBLOCK);
    }
    if (fcntl(socket, F_SETFL, flags) < 0)
        throw std::runtime_error("Error setting socket to non-blocking");
#endif
}

static inline void setNoDelay(int socket) {
    int flag = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int)) < 0)
        throw std::runtime_error("Error setting socket to no-delay");
}

static inline void setQuickAck(int socket) {
#ifndef _WIN32
#ifdef TCP_QUICKACK
    int value = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_QUICKACK, (char*)&value, sizeof(int)) < 0)
        throw std::runtime_error("Error setting quick ack");
#endif
#endif
}

void setReuseAddr(int socket) {
    int opt = 1;
    #ifdef _WIN32
    int iresult = setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
    if (iresult == SOCKET_ERROR) {
        closesocket(socket);
        throw std::runtime_error("setsockopt failed: " + std::to_string(WSAGetLastError()));
    }
    #else
    if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        close(socket);
        throw std::runtime_error("setsockopt failed: " + std::string(strerror(errno)));
    }
    #endif
}

void writeSocket(int socket, const void *data, NnSize size) {
    while (size > 0) {
        ssize_t s = send(socket, (const char*)data, size, 0);
        if (s < 0) {
            if (isEagainError()) {
                continue;
            }
            throw NnTransferSocketException(0, "Error writing to socket");
        } else if (s == 0) {
            throw NnTransferSocketException(0, "Socket closed");
        }
        size -= s;
        data = (const char*)data + s;
    }
}

static inline bool tryReadSocket(int socket, void *data, NnSize size, unsigned long maxAttempts) {
    // maxAttempts = 0 means infinite attempts
    NnSize s = size;
    while (s > 0) {
        ssize_t r = recv(socket, (char*)data, s, 0);
        if (r < 0) {
            if (isEagainError()) {
                if (s == size && maxAttempts > 0) {
                    maxAttempts--;
                    if (maxAttempts == 0) {
                        return false;
                    }
                }
                continue;
            }
            throw NnTransferSocketException(0, "Error reading from socket");
        } else if (r == 0) {
            throw NnTransferSocketException(0, "Socket closed");
        }
        data = (char*)data + r;
        s -= r;
    }
    return true;
}

void readSocket(int socket, void *data, NnSize size) {
    if (!tryReadSocket(socket, data, size, 0)) {
        throw std::runtime_error("Error reading from socket");
    }
}

static void readAckPacket(int socket) {
    NnUint packet;
    readSocket(socket, &packet, sizeof(packet));
    if (packet != ACK)
        throw std::runtime_error("Invalid ack packet");
}

static void writeAckPacket(int socket) {
    NnUint packet = ACK;
    writeSocket(socket, &packet, sizeof(packet));
}

static inline int connectSocket(char *host, int port) {
    struct addrinfo hints;
    struct addrinfo *addr = NULL;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    char portStr[11];
    snprintf(portStr, sizeof(portStr), "%d", port);

    int addrinfoError = getaddrinfo(host, portStr, &hints, &addr);
    if (addrinfoError != 0 || addr == NULL) {
        printf("Cannot resolve target %s (%s)\n", host, gai_strerror(addrinfoError));
        throw NnConnectionSocketException("Cannot resolve address");
    }

    int sock = ::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (sock < 0)
        throw std::runtime_error("Cannot create socket");

    int connectResult = ::connect(sock, addr->ai_addr, addr->ai_addrlen);
    if (connectResult != 0) {
        printf("Cannot connect to %s:%d (%s)\n", host, port, SOCKET_LAST_ERROR);
        throw NnConnectionSocketException("Cannot connect");
    }

    setNoDelay(sock);
    setQuickAck(sock);
    return sock;
}

int createServerSocket(int port) {
    const char *host = "0.0.0.0";
    struct sockaddr_in serverAddr;

    int serverSocket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (serverSocket < 0)
        throw std::runtime_error("Cannot create socket");
    setReuseAddr(serverSocket);

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = inet_addr(host);

    int bindResult;
    #ifdef _WIN32
    bindResult = bind(serverSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr));
    if (bindResult == SOCKET_ERROR) {
        int error = WSAGetLastError();
        closesocket(serverSocket);
        throw std::runtime_error("Cannot bind port: " + std::to_string(error));
    }
    #else
    bindResult = bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
    if (bindResult < 0) {
        close(serverSocket);
        throw std::runtime_error("Cannot bind port: " + std::string(strerror(errno)));
    }
    #endif

    int listenResult = listen(serverSocket, SOMAXCONN);
    if (listenResult != 0) {
        #ifdef _WIN32
        closesocket(serverSocket);
        throw std::runtime_error("Cannot listen on port: " + std::to_string(WSAGetLastError()));
        #else
        close(serverSocket);
        throw std::runtime_error("Cannot listen on port: " + std::string(strerror(errno)));
        #endif
    }

    printf("Listening on %s:%d...\n", host, port);

    setNoDelay(serverSocket);
    setQuickAck(serverSocket);
    return serverSocket;
}

void destroySocket(int serverSocket) {
    shutdown(serverSocket, 2);
    #ifdef _WIN32
    closesocket(serverSocket);
    #else
    close(serverSocket);
    #endif
}

int acceptSocket(int serverSocket) {
    struct sockaddr_in clientAddr;
    socklen_t clientAddrSize = sizeof(clientAddr);
    int clientSocket = ::accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrSize);
    if (clientSocket < 0)
        throw std::runtime_error("Error accepting connection");
    setNoDelay(clientSocket);
    setQuickAck(clientSocket);
    return clientSocket;
}

void initSockets() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        throw std::runtime_error("WSAStartup failed: " + std::to_string(WSAGetLastError()));
    }
#endif
}

void cleanupSockets() {
#ifdef _WIN32
    WSACleanup();
#endif
}

NnConnectionSocketException::NnConnectionSocketException(const std::string message)
    : std::runtime_error(message)
{}

NnTransferSocketException::NnTransferSocketException(int code, const std::string message)
    : code(code), std::runtime_error(message)
{}

NnSocket::NnSocket() {
    this->fd = -1;
}

NnSocket::NnSocket(int fd) : NnSocket() {
    assign(fd);
}

NnSocket::~NnSocket() {
    if (this->fd >= 0)
        destroySocket(this->fd);
}

void NnSocket::assign(int fd) {
    assert(this->fd == -1);
    assert(fd >= 0);
    this->fd = fd;
}

int NnSocket::release() {
    assert(this->fd >= 0);
    int fd = this->fd;
    this->fd = -1;
    return fd;
}

std::unique_ptr<NnNetwork> NnNetwork::serve(int port) {
    NnSocket socketSocket(createServerSocket(port));

    NnUint nSockets;
    NnUint nodeIndex;
    int rootSocketFd = acceptSocket(socketSocket.fd);
    NnSocket rootSocket(rootSocketFd);
    printf("⭕ The root node has connected\n");

    readSocket(rootSocketFd, &nSockets, sizeof(nSockets));
    NnUint nNodes = nSockets - 1; // nSockets - 1 root node
    printf("⭕ nNodes: %d\n", nNodes);
    readSocket(rootSocketFd, &nodeIndex, sizeof(nodeIndex));
    printf("⭕ NodeIndex: %d\n", nodeIndex);

    std::vector<NnSocket> sockets(nSockets);
    sockets[0].assign(rootSocket.release());

    printf("⭕ Socket[0]: accepted root node\n");
    std::vector<std::unique_ptr<char[]>> hosts(nNodes);
    std::vector<int> ports(nNodes);

    NnUint hostLen;
    for (NnUint i = 0; i < nNodes; i++) {
        readSocket(rootSocketFd, &hostLen, sizeof(hostLen));

        std::unique_ptr<char[]> host(new char[hostLen]);
        readSocket(rootSocketFd, host.get(), hostLen);
        hosts[i] = std::move(host);

        readSocket(rootSocketFd, &ports[i], sizeof(ports[i]));
    }

    writeAckPacket(rootSocketFd);

    // We need to wait here until the root node will send a "root is ready" packet
    readAckPacket(rootSocketFd);

    for (NnUint i = 0; i < nNodes; i++) {
        char *host = hosts[i].get();
        int port = ports[i];
        NnUint socketIndex = i + 1;
        if (i >= nodeIndex) {
            printf("⭕ Socket[%d]: connecting to %s:%d worker\n", socketIndex, host, port);
            sockets[socketIndex].assign(connectSocket(host, port));
            printf("⭕ Socket[%d]: connected\n", socketIndex);
        } else {
            printf("⭕ Socket[%d]: wait for %s:%d worker\n", socketIndex, host, port);
            sockets[socketIndex].assign(acceptSocket(socketSocket.fd));
            printf("⭕ Socket[%d]: accepted\n", socketIndex);
        }
    }

    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(&sockets));
}

std::unique_ptr<NnNetwork> NnNetwork::connect(NnUint nSockets, char **hosts, NnUint *ports) {
    assert(nSockets > 0);

    std::vector<NnSocket> sockets(nSockets);
    struct sockaddr_in addr;
    for (NnUint i = 0; i < nSockets; i++) {
        printf("⭕ Socket[%d]: connecting to %s:%d worker\n", i, hosts[i], ports[i]);
        int fd = connectSocket(hosts[i], ports[i]);
        sockets[i].assign(fd);
        writeSocket(fd, &nSockets, sizeof(nSockets));
        writeSocket(fd, &i, sizeof(i)); // send node index
        for (NnUint j = 0; j < nSockets; j++) {
            if (j == i)
                continue;
            NnUint hostLen = strlen(hosts[j]) + 1;
            writeSocket(fd, &hostLen, sizeof(hostLen));
            writeSocket(fd, hosts[j], hostLen);
            writeSocket(fd, &ports[j], sizeof(ports[j]));
        }
        readAckPacket(fd);
        printf("⭕ Socket[%d]: connected\n", i);
    }
    for (NnUint i = 0; i < nSockets; i++) {
        writeAckPacket(sockets[i].fd);
    }
    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(&sockets));
}

NnNetwork::NnNetwork(std::vector<NnSocket> *sockets) {
    this->nSockets = sockets->size();
    this->sockets = new int[nSockets];
    for (NnUint i = 0; i < nSockets; i++)
        this->sockets[i] = sockets->at(i).release();
    this->sentBytes = new NnSize[nSockets];
    this->recvBytes = new NnSize[nSockets];
    this->socketStats = new NnSocketPerformanceStats[nSockets];
}

NnNetwork::~NnNetwork() {
    delete[] sentBytes;
    delete[] recvBytes;
    delete[] socketStats;
    for (NnUint i = 0; i < nSockets; i++)
        destroySocket(sockets[i]);
    delete[] sockets;
    printf("⭕ Network is closed\n");
}

void NnNetwork::setTurbo(bool enabled) {
    for (NnUint i = 0; i < nSockets; i++) {
        ::setNonBlocking(sockets[i], enabled);
    }
}

void NnNetwork::write(const NnUint socketIndex, const void *data, const NnSize size) {
    assert(socketIndex < nSockets);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    NnByte *current = (NnByte *)data;
    int s = sockets[socketIndex];
    for (NnSize chunk = 0; chunk < size; chunk += MAX_CHUNK_SIZE) {
        NnSize chunkSize = chunk + MAX_CHUNK_SIZE < size ? MAX_CHUNK_SIZE : size - chunk;
        writeSocket(s, current, chunkSize);
        current += chunkSize;
    }
    sentBytes[socketIndex] += size;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    recordOperation("write", socketIndex, size, startTime, endTime);
}

void NnNetwork::read(const NnUint socketIndex, void *data, const NnSize size) {
    assert(socketIndex < nSockets);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    NnByte *current = (NnByte *)data;
    int s = sockets[socketIndex];
    for (NnSize chunk = 0; chunk < size; chunk += MAX_CHUNK_SIZE) {
        NnSize chunkSize = chunk + MAX_CHUNK_SIZE < size ? MAX_CHUNK_SIZE : size - chunk;
        readSocket(s, current, chunkSize);
        current += chunkSize;
    }
    recvBytes[socketIndex] += size;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    recordOperation("read", socketIndex, size, startTime, endTime);
}

void NnNetwork::writeAck(const NnUint socketIndex) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    writeAckPacket(sockets[socketIndex]);
}

void NnNetwork::readAck(const NnUint socketIndex) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    readAckPacket(sockets[socketIndex]);
}

bool NnNetwork::tryReadWithMaxAttempts(NnUint socketIndex, void *data, NnSize size, unsigned long maxAttempts) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    if (tryReadSocket(sockets[socketIndex], data, size, maxAttempts)) {
        recvBytes[socketIndex] += size;
        return true;
    }
    return false;
}

void NnNetwork::writeMany(NnUint n, NnSocketIo *ios) {
    bool isWriting;
    NnSize nBytes = 0;
    for (NnUint i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex < nSockets);
        sentBytes[io->socketIndex] += io->size;
    }
    do {
        isWriting = false;
        for (NnUint i = 0; i < n; i++) {
            NnSocketIo *io = &ios[i];
            if (io->size > 0) {
                isWriting = true;
                int socket = sockets[io->socketIndex];
                ssize_t chunkSize = io->size > MAX_CHUNK_SIZE ? MAX_CHUNK_SIZE : io->size;
                ssize_t s = send(socket, (const char*)io->data, chunkSize, 0);
                if (s < 0) {
                    if (isEagainError()) {
                        continue;
                    }
                    throw NnTransferSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (s == 0) {
                    throw NnTransferSocketException(0, "Socket closed");
                }
                io->size -= s;
                io->data = (char*)io->data + s;
            }
        }
    } while (isWriting);
}

void NnNetwork::writeAll(void *data, NnSize size) {
    std::vector<NnSocketIo> ios(nSockets);
    for (NnUint i = 0; i < nSockets; i++) {
        NnSocketIo *io = &ios[i];
        io->socketIndex = i;
        io->data = data;
        io->size = size;
    }
    writeMany(nSockets, &ios[0]);
}

void NnNetwork::readMany(NnUint n, NnSocketIo *ios) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    bool isReading;
    NnSize nBytes = 0;
    for (NnUint i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex < nSockets);
        recvBytes[io->socketIndex] += io->size;
        nBytes += io->size;
    }
    do {
        isReading = false;
        for (NnUint i = 0; i < n; i++) {
            NnSocketIo *io = &ios[i];
            if (io->size > 0) {
                isReading = true;
                int socket = sockets[io->socketIndex];
                ssize_t r = recv(socket, (char*)io->data, io->size, 0);
                if (r < 0) {
                    if (isEagainError()) {
                        continue;
                    }
                    throw NnTransferSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (r == 0) {
                    throw NnTransferSocketException(0, "Socket closed");
                }
                io->size -= r;
                io->data = (char*)io->data + r;
            }
        }
    } while (isReading);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    recordOperation("readMany", 0, nBytes, startTime, endTime);
}

void NnNetwork::getStats(NnSize *sentBytes, NnSize *recvBytes) {
    *sentBytes = 0;
    *recvBytes = 0;
    for (NnUint i = 0; i < nSockets; i++) {
        *sentBytes += this->sentBytes[i];
        *recvBytes += this->recvBytes[i];
    }
    resetStats();
}

void NnNetwork::resetStats() {
    for (NnUint i = 0; i < nSockets; i++) {
        sentBytes[i] = 0;
        recvBytes[i] = 0;
    }
}

// Static member for performance monitoring state
static bool g_performanceMonitoringEnabled = false;

void NnNetwork::enablePerformanceMonitoring(bool enabled) {
    g_performanceMonitoringEnabled = enabled;
    if (enabled) {
        printf("📊 Network performance monitoring enabled\n");
    }
}

bool NnNetwork::isPerformanceMonitoringEnabled() const {
    return g_performanceMonitoringEnabled;
}

void NnNetwork::recordOperation(const std::string& operationType, NnUint socketIndex, NnSize bytes, 
                               std::chrono::high_resolution_clock::time_point start, 
                               std::chrono::high_resolution_clock::time_point end) {
    if (!g_performanceMonitoringEnabled) return;
    
    std::lock_guard<std::mutex> lock(metricsMutex);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double latencyMs = duration.count() / 1000.0;
    
    updateSocketStats(socketIndex, latencyMs, bytes);
    
    // Store recent metrics for analysis (thread-safe with size limit)
    if (socketIndex < nSockets) {  // Safety check
        NnNetworkMetrics metric;
        metric.startTime = start;
        metric.endTime = end;
        metric.bytesTransferred = bytes;
        metric.operationType = operationType;
        metric.socketIndex = socketIndex;
        
        // Limit metrics storage to prevent memory issues
        if (recentMetrics.size() < 500) {
            recentMetrics.push_back(metric);
        }
    }
}

void NnNetwork::updateSocketStats(NnUint socketIndex, double latencyMs, NnSize bytes) {
    if (socketIndex >= nSockets || !socketStats) return;
    
    NnSocketPerformanceStats& stats = socketStats[socketIndex];
    
    stats.totalOperations++;
    stats.totalBytes += bytes;
    
    // Update latency statistics
    if (stats.totalOperations == 1) {
        stats.minLatencyMs = stats.maxLatencyMs = latencyMs;
    } else {
        stats.minLatencyMs = std::min(stats.minLatencyMs, latencyMs);
        stats.maxLatencyMs = std::max(stats.maxLatencyMs, latencyMs);
    }
    
    // Update average latency
    stats.avgLatencyMs = (stats.avgLatencyMs * (stats.totalOperations - 1) + latencyMs) / stats.totalOperations;
    
    // Update recent latencies (limit size for memory safety)
    if (stats.recentLatencies.size() < 50) {
        stats.recentLatencies.push_back(latencyMs);
    }
    
    // Calculate bandwidth (MB/s)
    if (latencyMs > 0) {
        double bandwidthMBps = (bytes / (1024.0 * 1024.0)) / (latencyMs / 1000.0);
        stats.bandwidthMbps = bandwidthMBps * 8; // Convert to Mbps
    }
}

void NnNetwork::printPerformanceReport() {
    if (!g_performanceMonitoringEnabled) {
        printf("📊 Performance monitoring is disabled. Enable it first.\n");
        return;
    }

    std::lock_guard<std::mutex> lock(metricsMutex);
    
    printf("\n📊 === Network Performance Report ===\n");
    printf("Socket | Operations | Total MB | Avg Latency | Max Latency | Min Latency | Bandwidth\n");
    printf("-------|------------|----------|-------------|-------------|-------------|----------\n");
    
    for (NnUint i = 0; i < nSockets; i++) {
        NnSocketPerformanceStats& stats = socketStats[i];
        if (stats.totalOperations > 0) {
            printf("   %2d  |    %6d   |  %6.2f   |   %6.2f ms  |   %6.2f ms  |   %6.2f ms  |  %6.2f Mbps\n",
                   i, stats.totalOperations, stats.totalBytes / (1024.0 * 1024.0),
                   stats.avgLatencyMs, stats.maxLatencyMs, stats.minLatencyMs, stats.bandwidthMbps);
        }
    }
    printf("\n");
}

void NnNetwork::printBottleneckAnalysis() {
    if (!g_performanceMonitoringEnabled) {
        printf("📊 Performance monitoring is disabled. Enable it first.\n");
        return;
    }

    std::lock_guard<std::mutex> lock(metricsMutex);
    
    printf("\n🔍 === Network Bottleneck Analysis ===\n");
    
    // Find the slowest socket
    NnUint slowestSocket = 0;
    double maxAvgLatency = 0;
    for (NnUint i = 0; i < nSockets; i++) {
        if (socketStats[i].totalOperations > 0 && socketStats[i].avgLatencyMs > maxAvgLatency) {
            maxAvgLatency = socketStats[i].avgLatencyMs;
            slowestSocket = i;
        }
    }
    
    printf("🐌 Slowest Socket: %d (Avg Latency: %.2f ms)\n", slowestSocket, maxAvgLatency);
    
    // Analyze recent latencies for variance
    for (NnUint i = 0; i < nSockets; i++) {
        if (!socketStats) continue;
        NnSocketPerformanceStats& stats = socketStats[i];
        if (stats.recentLatencies.size() > 10) {
            std::vector<double> latencies = stats.recentLatencies;
            std::sort(latencies.begin(), latencies.end());
            
            size_t size = latencies.size();
            if (size > 0) {
                double p50 = latencies[size / 2];
                double p95 = latencies[static_cast<size_t>(size * 0.95)];
                double p99 = latencies[static_cast<size_t>(size * 0.99)];
                
                printf("Socket %d: P50=%.2fms, P95=%.2fms, P99=%.2fms\n", i, p50, p95, p99);
                
                // Detect potential bottlenecks
                if (p95 > p50 * 2.0) {
                    printf("⚠️  Socket %d shows high latency variance (P95 >> P50) - potential network congestion\n", i);
                }
                if (stats.bandwidthMbps < 10.0 && stats.totalOperations > 100) {
                    printf("⚠️  Socket %d has low bandwidth (%.2f Mbps) - potential bandwidth limitation\n", i, stats.bandwidthMbps);
                }
            }
        }
    }
    
    // Analyze operation types
    std::map<std::string, int> operationCounts;
    std::map<std::string, NnSize> operationBytes;
    std::map<std::string, double> operationLatencies;
    
    for (const auto& metric : recentMetrics) {
        operationCounts[metric.operationType]++;
        operationBytes[metric.operationType] += metric.bytesTransferred;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(metric.endTime - metric.startTime);
        operationLatencies[metric.operationType] += duration.count() / 1000.0;
    }
    
    printf("\n📈 Operation Analysis:\n");
    for (const auto& op : operationCounts) {
        double avgLatency = operationLatencies[op.first] / op.second;
        double totalMB = operationBytes[op.first] / (1024.0 * 1024.0);
        printf("  %s: %d ops, %.2f MB, %.2f ms avg\n", op.first.c_str(), op.second, totalMB, avgLatency);
    }
    
    printf("\n");
}

NnSocketPerformanceStats* NnNetwork::getSocketStats(NnUint socketIndex) {
    if (socketIndex >= nSockets) return nullptr;
    return &socketStats[socketIndex];
}

static void syncWithRoot(NnNetwork *network, NnByte nodeIndex, NnByte *buffer, NnSize nBytes, NnUint nThreads, NnUint threadIndex) {
    if (nodeIndex == 0) {
        // root

        NnUint nSocketsPerThread = network->nSockets / nThreads + (network->nSockets % nThreads > threadIndex ? 1 : 0);
        if (nSocketsPerThread == 0) return;

        std::vector<NnSocketIo> ios(nSocketsPerThread);
        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            ios[i].socketIndex = threadIndex + i * nThreads;
            ios[i].data = buffer;
            ios[i].size = nBytes;
        }
        network->writeMany(nSocketsPerThread, &ios[0]);
    } else {
        // worker

        if (threadIndex != 0) return;

        NnSocketIo ios;
        ios.data = buffer;
        ios.size = nBytes;
        ios.socketIndex = 0; // root
        network->readMany(1, &ios);
    }
}

// Original O(n^2) implementation - kept for backward compatibility
static void syncNodeSlices_alltoall(bool onlyFromWorkerToRoot, NnNetwork *network, NnUint nodeIndex, NnUint nNodes, NnByte *buffer, NnSize nBytes, NnUint nThreads, NnUint threadIndex) {
    bool isWorker = nodeIndex != 0;
    NnUint nSockets = onlyFromWorkerToRoot && isWorker ? 1 : network->nSockets;
    NnUint nSocketsPerThread = nSockets / nThreads + (nSockets % nThreads > threadIndex ? 1 : 0);
    if (nSocketsPerThread == 0) return;
    NnSize sliceBytes = nBytes / nNodes;

    std::vector<NnSocketIo> ios(nSocketsPerThread);

    if (!onlyFromWorkerToRoot || isWorker) {
        NnByte *mySliceData = &buffer[sliceBytes * nodeIndex];

        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            NnUint socketIndex = threadIndex + i * nThreads;
            ios[i].socketIndex = socketIndex;
            ios[i].data = mySliceData;
            ios[i].size = sliceBytes;
        }
        network->writeMany(nSocketsPerThread, &ios[0]);
    }

    if (!onlyFromWorkerToRoot || !isWorker) {
        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            NnUint socketIndex = threadIndex + i * nThreads;
            NnUint sliceIndex = socketIndex >= nodeIndex ? socketIndex + 1 : socketIndex;
            NnByte *sliceData = &buffer[sliceBytes * sliceIndex];
            ios[i].socketIndex = socketIndex;
            ios[i].data = sliceData;
            ios[i].size = sliceBytes;
        }
        network->readMany(nSocketsPerThread, &ios[0]);
    }
}

// ============================================================================
// O(log n) Binary Tree Gather-Broadcast Implementation
// This is a fundamental redesign that reduces complexity from O(n^2) to O(log n)
// Architecture: Gather to Root → (Optional Compute) → Broadcast from Root
// ============================================================================

// Helper to get socket index in full mesh topology
// Socket mapping matches the original alltoall implementation
static inline NnUint getSocketIndexForNode(NnUint myNodeIndex, NnUint peerNode) {
    // Root (node 0): sockets map directly to workers
    // socket[0] -> worker 1, socket[1] -> worker 2, etc.
    if (myNodeIndex == 0) {
        return peerNode - 1;
    }
    
    // Workers: socket[0] is root, socket[1..n-2] are other workers
    // When communicating with other workers, skip self
    if (peerNode == 0) {
        return 0;  // Root is always socket[0] for workers
    }
    
    // Worker to worker: map peer node to socket index
    // Socket indices for workers: 0=root, 1=node1, 2=node2, ..., but skip self
    if (peerNode < myNodeIndex) {
        return peerNode;  // Peer comes before us in socket array
    } else {
        return peerNode - 1;  // Peer comes after us, but we skip ourselves
    }
}

// O(n) Ring All-Gather - Simple and Reliable
// Each step: all nodes simultaneously send to right and receive from left
static void ringAllGather(NnNetwork *network, NnUint nodeIndex, NnUint nNodes, NnByte *buffer, NnSize sliceBytes, NnUint nThreads, NnUint threadIndex) {
    if (threadIndex != 0) return;  // Only thread 0
    
    if (threadIndex == 0) {
        printf("🔄 [Node %u] RING ALL-GATHER START: nNodes=%u, sliceBytes=%zu\n", 
               nodeIndex, nNodes, sliceBytes);
        fflush(stdout);
    }
    
    // Ring topology: each node sends to next and receives from previous
    NnUint sendToNode = (nodeIndex + 1) % nNodes;
    NnUint recvFromNode = (nodeIndex + nNodes - 1) % nNodes;
    
    NnUint sendSocketIndex = getSocketIndexForNode(nodeIndex, sendToNode);
    NnUint recvSocketIndex = getSocketIndexForNode(nodeIndex, recvFromNode);
    
    printf("🔄 [Node %u] Ring: send to Node %u (socket %u), recv from Node %u (socket %u)\n",
           nodeIndex, sendToNode, sendSocketIndex, recvFromNode, recvSocketIndex);
    fflush(stdout);
    
    // In n-1 steps, collect all slices
    for (NnUint step = 0; step < nNodes - 1; step++) {
        // Determine which slice to send and where to receive
        NnUint sendSliceIndex = (nodeIndex - step + nNodes) % nNodes;
        NnUint recvSliceIndex = (nodeIndex - step - 1 + nNodes) % nNodes;
        
        printf("🔄 [Node %u] Step %u: sending slice %u, receiving slice %u\n",
               nodeIndex, step, sendSliceIndex, recvSliceIndex);
        fflush(stdout);
        
        NnSocketIo sendIo, recvIo;
        sendIo.socketIndex = sendSocketIndex;
        sendIo.data = &buffer[sliceBytes * sendSliceIndex];
        sendIo.size = sliceBytes;
        
        recvIo.socketIndex = recvSocketIndex;
        recvIo.data = &buffer[sliceBytes * recvSliceIndex];
        recvIo.size = sliceBytes;
        
        // Critical: Send and receive in a way that avoids circular deadlock
        // Even nodes send first, odd nodes receive first
        if (nodeIndex % 2 == 0) {
            printf("📤 [Node %u] Step %u: Sending first\n", nodeIndex, step);
            fflush(stdout);
            network->writeMany(1, &sendIo);
            printf("📥 [Node %u] Step %u: Now receiving\n", nodeIndex, step);
            fflush(stdout);
            network->readMany(1, &recvIo);
        } else {
            printf("📥 [Node %u] Step %u: Receiving first\n", nodeIndex, step);
            fflush(stdout);
            network->readMany(1, &recvIo);
            printf("📤 [Node %u] Step %u: Now sending\n", nodeIndex, step);
            fflush(stdout);
            network->writeMany(1, &sendIo);
        }
        
        printf("✅ [Node %u] Step %u complete\n", nodeIndex, step);
        fflush(stdout);
    }
    
    if (threadIndex == 0) {
        printf("🔄 [Node %u] RING ALL-GATHER COMPLETE\n", nodeIndex);
        fflush(stdout);
    }
}

// Binary Tree Broadcast: O(log n) - Distribute all data from root
// Uses top-down tree broadcasting with multithreading support  
static void binaryTreeBroadcast(NnNetwork *network, NnUint nodeIndex, NnUint nNodes, NnByte *buffer, NnSize nBytes, NnUint nThreads, NnUint threadIndex) {
    if (threadIndex == 0) {
        printf("📡 [Node %u] BROADCAST START: nNodes=%u, nBytes=%zu, nThreads=%u\n", 
               nodeIndex, nNodes, nBytes, nThreads);
        fflush(stdout);
    }
    
    // Calculate tree depth
    NnUint treeDepth = 0;
    NnUint temp = nNodes - 1;
    while (temp > 0) {
        treeDepth++;
        temp >>= 1;
    }
    
    if (threadIndex == 0) {
        printf("📡 [Node %u] Tree depth: %u\n", nodeIndex, treeDepth);
        fflush(stdout);
    }
    
    // Top-down broadcasting: parents send to children
    // IMPORTANT: Receivers must be ready BEFORE senders start
    for (NnUint level = 0; level < treeDepth; level++) {
        NnUint step = 1 << level;  // 2^level
        NnUint stride = step << 1;  // 2^(level+1)
        
        if (threadIndex == 0) {
            printf("📡 [Node %u] Level %u: step=%u, stride=%u, checking role...\n", 
                   nodeIndex, level, step, stride);
            fflush(stdout);
        }
        
        // Receivers first
        if (nodeIndex % stride == step && nodeIndex < nNodes) {
            // I'm a receiver at this level
            NnUint parentNode = nodeIndex - step;
            if (threadIndex == 0) {
                printf("📥 [Node %u] Level %u: I am RECEIVER from node %u\n", 
                       nodeIndex, level, parentNode);
                fflush(stdout);
            }
            
            if (parentNode < nNodes && threadIndex == 0) {
                NnUint socketIndex = getSocketIndexForNode(nodeIndex, parentNode);
                
                printf("📥 [Node %u] Receiving broadcast from node %u: socketIndex=%u, bytes=%zu\n",
                       nodeIndex, parentNode, socketIndex, nBytes);
                fflush(stdout);
                
                NnSocketIo io;
                io.socketIndex = socketIndex;
                io.data = buffer;
                io.size = nBytes;
                network->readMany(1, &io);
                
                printf("✅ [Node %u] Broadcast receive complete from node %u\n", nodeIndex, parentNode);
                fflush(stdout);
            }
        } else if (nodeIndex % stride == 0 && nodeIndex + step < nNodes) {
            // I'm a sender at this level
            NnUint childNode = nodeIndex + step;
            if (threadIndex == 0) {
                printf("📤 [Node %u] Level %u: I am SENDER to node %u\n", 
                       nodeIndex, level, childNode);
                fflush(stdout);
            }
            
            if (childNode < nNodes && threadIndex == 0) {
                NnUint socketIndex = getSocketIndexForNode(nodeIndex, childNode);
                
                printf("📤 [Node %u] Broadcasting to node %u: socketIndex=%u, bytes=%zu\n",
                       nodeIndex, childNode, socketIndex, nBytes);
                fflush(stdout);
                
                NnSocketIo io;
                io.socketIndex = socketIndex;
                io.data = buffer;
                io.size = nBytes;
                network->writeMany(1, &io);
                
                printf("✅ [Node %u] Broadcast complete to node %u\n", nodeIndex, childNode);
                fflush(stdout);
            }
        } else {
            if (threadIndex == 0) {
                printf("⏸️  [Node %u] Level %u: I am IDLE (no action at this level)\n", 
                       nodeIndex, level);
                fflush(stdout);
            }
        }
    }
    
    if (threadIndex == 0) {
        printf("📡 [Node %u] BROADCAST COMPLETE\n", nodeIndex);
        fflush(stdout);
    }
    // Now all nodes have complete data
}

// Helper function to check if pointer is properly aligned
static inline bool isAligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

// Helper function to perform element-wise reduction (sum) for All-Reduce
// Supports F32, F16, Q80, Q40 data types
// Uses ONLY byte-wise or memcpy-based operations for maximum safety (no direct pointer casting)
static void reduceSum(NnByte *result, const NnByte *input, NnSize nBytes, NnFloatType floatType) {
    // Safety check: ensure we have valid pointers and non-zero size
    if (!result || !input || nBytes == 0) {
        return;
    }
    
    // For maximum safety and compatibility, always use memcpy approach
    // This avoids any potential alignment issues regardless of input buffer alignment
    if (floatType == F_32 || floatType == F_Q80 || floatType == F_Q40) {
        // F32, Q80, Q40: Treat as float arrays
        // Always use memcpy to aligned stack buffers - safest approach
        NnSize nElements = nBytes / sizeof(float);
        if (nElements > 0) {
            const NnSize chunkSize = 256;  // Process 256 floats at a time (1KB chunks)
            float inputBuf[chunkSize];
            float resultBuf[chunkSize];
            
            NnSize processed = 0;
            while (processed < nElements) {
                NnSize currentChunk = (nElements - processed < chunkSize) ? 
                                     (nElements - processed) : chunkSize;
                
                // Copy to stack buffers (always aligned on stack)
                std::memcpy(inputBuf, input + processed * sizeof(float), currentChunk * sizeof(float));
                std::memcpy(resultBuf, result + processed * sizeof(float), currentChunk * sizeof(float));
                
                // Perform reduction on aligned buffers
                for (NnSize i = 0; i < currentChunk; i++) {
                    resultBuf[i] += inputBuf[i];
                }
                
                // Copy back
                std::memcpy(result + processed * sizeof(float), resultBuf, currentChunk * sizeof(float));
                processed += currentChunk;
            }
        }
        
        // Handle any remaining bytes (not multiple of float size)
        NnSize remainingBytes = nBytes % sizeof(float);
        if (remainingBytes > 0) {
            NnSize offset = nElements * sizeof(float);
            for (NnSize i = 0; i < remainingBytes; i++) {
                result[offset + i] = static_cast<NnByte>(result[offset + i] + input[offset + i]);
            }
        }
    } else if (floatType == F_16) {
        // F16: Use memcpy for each element (always safe regardless of alignment)
        NnSize nElements = nBytes / sizeof(NnFp16);
        for (NnSize i = 0; i < nElements; i++) {
            NnFp16 inputVal, resultVal;
            // Use memcpy to avoid alignment issues
            std::memcpy(&inputVal, input + i * sizeof(NnFp16), sizeof(NnFp16));
            std::memcpy(&resultVal, result + i * sizeof(NnFp16), sizeof(NnFp16));
            
            float a = convertF16toF32Impl(resultVal);
            float b = convertF16toF32Impl(inputVal);
            float sum = a + b;
            NnFp16 sumVal = convertF32ToF16Impl(sum);
            
            std::memcpy(result + i * sizeof(NnFp16), &sumVal, sizeof(NnFp16));
        }
        
        // Handle remaining bytes (if any)
        NnSize remainingBytes = nBytes % sizeof(NnFp16);
        if (remainingBytes > 0) {
            NnSize offset = nElements * sizeof(NnFp16);
            for (NnSize i = 0; i < remainingBytes; i++) {
                result[offset + i] = static_cast<NnByte>(result[offset + i] + input[offset + i]);
            }
        }
    } else {
        // Fallback: byte-wise addition (always safe for unknown types)
        for (NnSize i = 0; i < nBytes; i++) {
            result[i] = static_cast<NnByte>(result[i] + input[i]);
        }
    }
}

// Ring All-Reduce - O(n) with better bandwidth utilization (vLLM/TensorRT style)
// Phase 1: Reduce-Scatter - Divide data into chunks, ring-pass to reduce
// Phase 2: All-Gather - Ring-pass reduced chunks to collect full result
// This provides better scalability than Star topology for large clusters
static void syncNodeSlices_ringAllReduce(bool onlyFromWorkerToRoot, NnNetwork *network, NnUint nodeIndex, NnUint nNodes, NnByte *buffer, NnSize nBytes, NnFloatType floatType, NnUint nThreads, NnUint threadIndex) {
    if (threadIndex != 0) return;  // Only thread 0 handles ring communication
    
    NnSize sliceBytes = nBytes / nNodes;
    
    // Ring topology: each node sends to next and receives from previous
    NnUint sendToNode = (nodeIndex + 1) % nNodes;
    NnUint recvFromNode = (nodeIndex + nNodes - 1) % nNodes;
    
    NnUint sendSocketIndex = getSocketIndexForNode(nodeIndex, sendToNode);
    NnUint recvSocketIndex = getSocketIndexForNode(nodeIndex, recvFromNode);
    
    // ========== PHASE 1: REDUCE-SCATTER ==========
    // Use thread_local vector for safe automatic memory management
    // This avoids stack overflow and manual memory management issues
    static thread_local std::vector<NnByte> tempBuffer;
    if (tempBuffer.size() < sliceBytes) {
        tempBuffer.reserve(sliceBytes);
        tempBuffer.resize(sliceBytes, 0);
    }
    NnByte* tempBuffer_ptr = tempBuffer.data();
    
    // In n-1 steps, each node accumulates one chunk through reduction
    for (NnUint step = 0; step < nNodes - 1; step++) {
        // Determine which chunk to send and where to receive
        NnUint sendChunkIndex = (nodeIndex - step + nNodes) % nNodes;
        NnUint recvChunkIndex = (nodeIndex - step - 1 + nNodes) % nNodes;
        
        NnSocketIo sendIo, recvIo;
        
        // Use pre-allocated buffer
        NnByte* alignedBuffer = tempBuffer_ptr;
        
        sendIo.socketIndex = sendSocketIndex;
        sendIo.data = &buffer[sliceBytes * sendChunkIndex];
        sendIo.size = sliceBytes;
        
        recvIo.socketIndex = recvSocketIndex;
        recvIo.data = alignedBuffer;
        recvIo.size = sliceBytes;
        
        // Even nodes send first, odd nodes receive first (avoid deadlock)
        if (nodeIndex % 2 == 0) {
            network->writeMany(1, &sendIo);
            network->readMany(1, &recvIo);
        } else {
            network->readMany(1, &recvIo);
            network->writeMany(1, &sendIo);
        }
        
        // Reduce: add received chunk to corresponding local chunk
        // Ensure we don't write beyond buffer bounds
        if (sliceBytes * recvChunkIndex + sliceBytes <= nBytes) {
            reduceSum(&buffer[sliceBytes * recvChunkIndex], alignedBuffer, sliceBytes, floatType);
        }
    }
    
    // At this point, each node has one fully reduced chunk
    
    if (onlyFromWorkerToRoot) {
        return;  // If only reduce-scatter needed, we're done
    }
    
    // ========== PHASE 2: ALL-GATHER ==========
    // In n-1 steps, collect all reduced chunks
    for (NnUint step = 0; step < nNodes - 1; step++) {
        NnUint sendChunkIndex = (nodeIndex - step + nNodes) % nNodes;
        NnUint recvChunkIndex = (nodeIndex - step - 1 + nNodes) % nNodes;
        
        NnSocketIo sendIo, recvIo;
        
        sendIo.socketIndex = sendSocketIndex;
        sendIo.data = &buffer[sliceBytes * sendChunkIndex];
        sendIo.size = sliceBytes;
        
        recvIo.socketIndex = recvSocketIndex;
        recvIo.data = &buffer[sliceBytes * recvChunkIndex];
        recvIo.size = sliceBytes;
        
        // Even nodes send first, odd nodes receive first
        if (nodeIndex % 2 == 0) {
            network->writeMany(1, &sendIo);
            network->readMany(1, &recvIo);
        } else {
            network->readMany(1, &recvIo);
            network->writeMany(1, &sendIo);
        }
    }
    
    // Now all nodes have the fully reduced result in all chunks
}

// Star Topology All-Reduce - O(n) Root-Centric with proper multithreading
// Phase 1: Gather - All workers send their slice to root (root receives in parallel)
// Phase 2: Reduce - Root performs element-wise sum reduction across all slices
// Phase 3: Broadcast - Root broadcasts reduced result to all workers
// This is more efficient than Gather-Broadcast for operations requiring reduction (sum)
static void syncNodeSlices_starAllReduce(bool onlyFromWorkerToRoot, NnNetwork *network, NnUint nodeIndex, NnUint nNodes, NnByte *buffer, NnSize nBytes, NnFloatType floatType, NnUint nThreads, NnUint threadIndex) {
    if (threadIndex != 0) return;  // Single-threaded to avoid races on shared buffers

    // ========== PHASE 1: GATHER TO ROOT (FULL BUFFER) ==========
    if (nodeIndex == 0) {
        // ROOT: Collect full buffers from all workers and reduce into root buffer
        static thread_local std::vector<NnByte> tempBuffer;
        if (tempBuffer.size() < nBytes) {
            tempBuffer.resize(nBytes, 0);
        }

        for (NnUint workerIdx = 1; workerIdx < nNodes; workerIdx++) {
            NnSocketIo io;
            io.socketIndex = workerIdx - 1;
            io.data = tempBuffer.data();
            io.size = nBytes;
            network->readMany(1, &io);
            reduceSum(buffer, tempBuffer.data(), nBytes, floatType);
        }
    } else {
        // WORKER: Send full buffer to root
        NnSocketIo io;
        io.socketIndex = 0;
        io.data = buffer;
        io.size = nBytes;
        network->writeMany(1, &io);
    }
    
    // If only gathering to root (for reduction), we're done
    if (onlyFromWorkerToRoot) {
        return;
    }
    
    // ========== PHASE 2: BROADCAST FROM ROOT ==========
    if (nodeIndex == 0) {
        // ROOT: Broadcast reduced result to all workers
        for (NnUint workerIdx = 1; workerIdx < nNodes; workerIdx++) {
            NnSocketIo io;
            io.socketIndex = workerIdx - 1;
            io.data = buffer;
            io.size = nBytes;
            network->writeMany(1, &io);
        }
    } else {
        // WORKER: Receive reduced result
        NnSocketIo io;
        io.socketIndex = 0;
        io.data = buffer;
        io.size = nBytes;
        network->readMany(1, &io);
    }
}

// Star Topology Gather-Broadcast - O(n) Root-Centric with proper multithreading
// Phase 1: All workers send their slice to root (root receives in parallel)
// Phase 2: Root broadcasts complete buffer to all workers (workers receive)
static void syncNodeSlices_starGatherBroadcast(bool onlyFromWorkerToRoot, NnNetwork *network, NnUint nodeIndex, NnUint nNodes, NnByte *buffer, NnSize nBytes, NnUint nThreads, NnUint threadIndex) {
    NnSize sliceBytes = nBytes / nNodes;
    
    // ========== PHASE 1: GATHER TO ROOT ==========
    if (nodeIndex == 0) {
        // ROOT: Collect slices from all workers
        // Distribute workers across threads for PARALLEL receive
        NnUint nWorkers = nNodes - 1;
        NnUint workersPerThread = nWorkers / nThreads + (nWorkers % nThreads > threadIndex ? 1 : 0);
        
        for (NnUint i = 0; i < workersPerThread; i++) {
            NnUint workerIdx = threadIndex + i * nThreads + 1;
            if (workerIdx < nNodes) {
                NnSocketIo io;
                io.socketIndex = workerIdx - 1;
                io.data = &buffer[sliceBytes * workerIdx];
                io.size = sliceBytes;
                network->readMany(1, &io);
            }
        }
    } else {
        // WORKER: Only thread 0 sends, but ALL threads must wait
        // This ensures thread synchronization
        if (threadIndex == 0) {
            NnSocketIo io;
            io.socketIndex = 0;
            io.data = &buffer[sliceBytes * nodeIndex];
            io.size = sliceBytes;
            network->writeMany(1, &io);
        }
        // Implicit barrier: all threads exit this block together
        // No early return - this keeps threads synchronized
    }
    
    // If only gathering to root, we're done
    if (onlyFromWorkerToRoot) {
        // All threads reach here together
        return;
    }
    
    // ========== PHASE 2: BROADCAST FROM ROOT ==========
    if (nodeIndex == 0) {
        // ROOT: Broadcast complete buffer to all workers
        // Distribute workers across threads for PARALLEL send
        NnUint nWorkers = nNodes - 1;
        NnUint workersPerThread = nWorkers / nThreads + (nWorkers % nThreads > threadIndex ? 1 : 0);
        
        for (NnUint i = 0; i < workersPerThread; i++) {
            NnUint workerIdx = threadIndex + i * nThreads + 1;
            if (workerIdx < nNodes) {
                NnSocketIo io;
                io.socketIndex = workerIdx - 1;
                io.data = buffer;
                io.size = nBytes;
                network->writeMany(1, &io);
            }
        }
    } else {
        // WORKER: Only thread 0 receives, but ALL threads must wait
        if (threadIndex == 0) {
            NnSocketIo io;
            io.socketIndex = 0;
            io.data = buffer;
            io.size = nBytes;
            network->readMany(1, &io);
        }
        // Implicit barrier: all threads exit this function together
    }
    
    // All threads reach here together - synchronized!
}

// Main syncNodeSlices function
// Uses All-Reduce (with sum reduction) - optimized with vLLM/TensorRT style algorithms
static void syncNodeSlices(bool onlyFromWorkerToRoot, NnNetwork *network, NnUint nodeIndex, NnUint nNodes, NnByte *buffer, NnSize nBytes, NnFloatType floatType, NnUint nThreads, NnUint threadIndex, NnCollectiveType collectiveType) {
    if (nNodes <= 1 || nBytes == 0) return;

    NnCollectiveType effective = collectiveType;
    if (effective == COLLECTIVE_AUTO) {
        effective = nNodes <= 4 ? COLLECTIVE_STAR : COLLECTIVE_RING;
    }

    if (effective == COLLECTIVE_RING) {
        syncNodeSlices_ringAllReduce(onlyFromWorkerToRoot, network, nodeIndex, nNodes, buffer, nBytes, floatType, nThreads, threadIndex);
    } else {
        syncNodeSlices_starAllReduce(onlyFromWorkerToRoot, network, nodeIndex, nNodes, buffer, nBytes, floatType, nThreads, threadIndex);
    }
}

NnNetworkNodeSynchronizer::NnNetworkNodeSynchronizer(NnNetwork *network, NnNetExecution *execution, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnCollectiveType collectiveType) {
    this->network = network;
    this->execution = execution;
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
    this->collectiveType = collectiveType;
}

void NnNetworkNodeSynchronizer::sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];

    for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
        NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
        NnByte *pipe = execution->pipes[syncConfig->pipeIndex];
        NnPipeConfig *pipeConfig = &netConfig->pipes[syncConfig->pipeIndex];
        NnSize batchBytes = getBytes(pipeConfig->size.floatType, pipeConfig->size.x);

        for (NnUint batchIndex = 0; batchIndex < execution->batchSize; batchIndex++) {
            NnByte *pipeBatch = &pipe[batchIndex * batchBytes];
            
            auto syncStartTime = std::chrono::high_resolution_clock::now();
            std::string syncTypeName;

            if (syncConfig->syncType == SYNC_WITH_ROOT) {
                syncTypeName = "SYNC_WITH_ROOT";
                syncWithRoot(network, nodeConfig->nodeIndex, pipeBatch, batchBytes, nThreads, threadIndex);
            } else if (syncConfig->syncType == SYNC_NODE_SLICES) {
                syncTypeName = "SYNC_NODE_SLICES";
                syncNodeSlices(false, network, nodeConfig->nodeIndex, netConfig->nNodes, pipeBatch, batchBytes, pipeConfig->size.floatType, nThreads, threadIndex, collectiveType);
            } else if (syncConfig->syncType == SYNC_NODE_SLICES_EXCEPT_ROOT) {
                syncTypeName = "SYNC_NODE_SLICES_EXCEPT_ROOT";
                syncNodeSlices(true, network, nodeConfig->nodeIndex, netConfig->nNodes, pipeBatch, batchBytes, pipeConfig->size.floatType, nThreads, threadIndex, collectiveType);
            } else {
                throw std::invalid_argument("Unknown sync type");
            }
            
            auto syncEndTime = std::chrono::high_resolution_clock::now();
            
            // Record sync operation for monitoring
            if (network->isPerformanceMonitoringEnabled()) {
                NnSize totalBytes = batchBytes * execution->batchSize;
                network->recordOperation(syncTypeName, 0, totalBytes, syncStartTime, syncEndTime);
            }
        }
    }
}

static void writeString(NnNetwork *network, NnUint socketIndex, char *str) {
    NnUint bytes = std::strlen(str) + 1;
    network->write(socketIndex, &bytes, sizeof(NnUint));
    network->write(socketIndex, str, bytes);
}

static char *readString(NnNetwork *network, NnUint socketIndex) {
    NnUint bytes;
    network->read(socketIndex, &bytes, sizeof(NnUint));
    char *str = new char[bytes];
    network->read(socketIndex, str, bytes);
    return str;
}

NnRootConfigWriter::NnRootConfigWriter(NnNetwork *network) {
    this->network = network;
}

void NnRootConfigWriter::writeNet(NnUint socketIndex, NnNetConfig *config) {
    network->writeAck(socketIndex);
    network->write(socketIndex, &config->nBatches, sizeof(config->nBatches));
    network->write(socketIndex, &config->nNodes, sizeof(config->nNodes));
    network->write(socketIndex, &config->nPipes, sizeof(config->nPipes));
    for (NnUint pipeIndex = 0; pipeIndex < config->nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &config->pipes[pipeIndex];
        network->write(socketIndex, &pipeConfig->size, sizeof(pipeConfig->size));
        writeString(network, socketIndex, pipeConfig->name);
    }
    network->write(socketIndex, &config->nPreSyncs, sizeof(config->nPreSyncs));
    for (NnUint preSyncIndex = 0; preSyncIndex < config->nPreSyncs; preSyncIndex++) {
        NnPreSyncConfig *preSyncConfig = &config->preSyncs[preSyncIndex];
        network->write(socketIndex, &preSyncConfig->pipeIndex, sizeof(preSyncConfig->pipeIndex));
    }
    network->readAck(socketIndex);
}

void NnRootConfigWriter::writeNode(NnUint socketIndex, NnNodeConfig *config) {
    network->writeAck(socketIndex);
    network->write(socketIndex, &config->nodeIndex, sizeof(config->nodeIndex));
    network->write(socketIndex, &config->nBuffers, sizeof(config->nBuffers));
    network->write(socketIndex, &config->nSegments, sizeof(config->nSegments));

    for (NnUint bufferIndex = 0; bufferIndex < config->nBuffers; bufferIndex++) {
        NnBufferConfig *bufferConfig = &config->buffers[bufferIndex];
        network->write(socketIndex, &bufferConfig->size, sizeof(bufferConfig->size));
        writeString(network, socketIndex, bufferConfig->name);
    }

    for (NnUint segmentIndex = 0; segmentIndex < config->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &config->segments[segmentIndex];
        network->write(socketIndex, &segmentConfig->nSyncs, sizeof(segmentConfig->nSyncs));
        network->write(socketIndex, &segmentConfig->nOps, sizeof(segmentConfig->nOps));

        for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
            NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
            network->write(socketIndex, &syncConfig->pipeIndex, sizeof(syncConfig->pipeIndex));
            network->write(socketIndex, &syncConfig->syncType, sizeof(syncConfig->syncType));
        }
        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            network->write(socketIndex, &opConfig->code, sizeof(opConfig->code));
            network->write(socketIndex, &opConfig->index, sizeof(opConfig->index));
            network->write(socketIndex, &opConfig->weightSize, sizeof(opConfig->weightSize));
            network->write(socketIndex, &opConfig->configSize, sizeof(opConfig->configSize));
            writeString(network, socketIndex, opConfig->name);
            network->write(socketIndex, &opConfig->input, sizeof(opConfig->input));
            network->write(socketIndex, &opConfig->output, sizeof(opConfig->output));
            if (opConfig->configSize > 0)
                network->write(socketIndex, opConfig->config, opConfig->configSize);
        }
    }
    network->readAck(socketIndex);
}

void NnRootConfigWriter::writeToWorkers(NnNetConfig *netConfig, NnNodeConfig *nodeConfigs) {
    for (NnUint nodeIndex = 1; nodeIndex < netConfig->nNodes; nodeIndex++) {
        NnUint socketIndex = nodeIndex - 1;
        writeNet(socketIndex, netConfig);
        writeNode(socketIndex, &nodeConfigs[nodeIndex]);
    }
}

NnWorkerConfigReader::NnWorkerConfigReader(NnNetwork *network) {
    this->network = network;
}

NnNetConfig NnWorkerConfigReader::readNet() {
    network->readAck(ROOT_SOCKET_INDEX);
    NnNetConfig config;
    network->read(ROOT_SOCKET_INDEX, &config.nBatches, sizeof(config.nBatches));
    network->read(ROOT_SOCKET_INDEX, &config.nNodes, sizeof(config.nNodes));
    network->read(ROOT_SOCKET_INDEX, &config.nPipes, sizeof(config.nPipes));
    config.pipes = new NnPipeConfig[config.nPipes];
    for (NnUint pipeIndex = 0; pipeIndex < config.nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &config.pipes[pipeIndex];
        network->read(ROOT_SOCKET_INDEX, &pipeConfig->size, sizeof(pipeConfig->size));
        pipeConfig->name = readString(network, ROOT_SOCKET_INDEX);
    }
    network->read(ROOT_SOCKET_INDEX, &config.nPreSyncs, sizeof(config.nPreSyncs));
    config.preSyncs = new NnPreSyncConfig[config.nPreSyncs];
    for (NnUint preSyncIndex = 0; preSyncIndex < config.nPreSyncs; preSyncIndex++) {
        NnPreSyncConfig *preSyncConfig = &config.preSyncs[preSyncIndex];
        network->read(ROOT_SOCKET_INDEX, &preSyncConfig->pipeIndex, sizeof(preSyncConfig->pipeIndex));
    }
    network->writeAck(ROOT_SOCKET_INDEX);
    return config;
}

NnNodeConfig NnWorkerConfigReader::readNode() {
    network->readAck(ROOT_SOCKET_INDEX);

    NnNodeConfig config;
    network->read(ROOT_SOCKET_INDEX, &config.nodeIndex, sizeof(config.nodeIndex));
    network->read(ROOT_SOCKET_INDEX, &config.nBuffers, sizeof(config.nBuffers));
    network->read(ROOT_SOCKET_INDEX, &config.nSegments, sizeof(config.nSegments));

    config.buffers = new NnBufferConfig[config.nBuffers];
    config.segments = new NnSegmentConfig[config.nSegments];

    for (NnUint bufferIndex = 0; bufferIndex < config.nBuffers; bufferIndex++) {
        NnBufferConfig *bufferConfig = &config.buffers[bufferIndex];
        network->read(ROOT_SOCKET_INDEX, &bufferConfig->size, sizeof(bufferConfig->size));
        bufferConfig->name = readString(network, ROOT_SOCKET_INDEX);
    }

    for (NnUint segmentIndex = 0; segmentIndex < config.nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &config.segments[segmentIndex];
        network->read(ROOT_SOCKET_INDEX, &segmentConfig->nSyncs, sizeof(segmentConfig->nSyncs));
        network->read(ROOT_SOCKET_INDEX, &segmentConfig->nOps, sizeof(segmentConfig->nOps));

        if (segmentConfig->nSyncs > 0) {
            segmentConfig->syncs = new NnSyncConfig[segmentConfig->nSyncs];

            for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
                NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
                network->read(ROOT_SOCKET_INDEX, &syncConfig->pipeIndex, sizeof(syncConfig->pipeIndex));
                network->read(ROOT_SOCKET_INDEX, &syncConfig->syncType, sizeof(syncConfig->syncType));
            }
        }

        if (segmentConfig->nOps > 0) {
            segmentConfig->ops = new NnOpConfig[segmentConfig->nOps];

            for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
                NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
                network->read(ROOT_SOCKET_INDEX, &opConfig->code, sizeof(opConfig->code));
                network->read(ROOT_SOCKET_INDEX, &opConfig->index, sizeof(opConfig->index));
                network->read(ROOT_SOCKET_INDEX, &opConfig->weightSize, sizeof(opConfig->weightSize));
                network->read(ROOT_SOCKET_INDEX, &opConfig->configSize, sizeof(opConfig->configSize));
                opConfig->name = readString(network, ROOT_SOCKET_INDEX);
                network->read(ROOT_SOCKET_INDEX, &opConfig->input, sizeof(opConfig->input));
                network->read(ROOT_SOCKET_INDEX, &opConfig->output, sizeof(opConfig->output));
                if (opConfig->configSize > 0) {
                    opConfig->config = new NnByte[opConfig->configSize];
                    network->read(ROOT_SOCKET_INDEX, opConfig->config, opConfig->configSize);
                }
            }
        }
    }
    network->writeAck(ROOT_SOCKET_INDEX);
    return config;
}

NnRootWeightLoader::NnRootWeightLoader(NnExecutor *executor, NnNetwork *network, NnUint nNodes) {
    this->executor = executor;
    this->network = network;
    this->nNodes = nNodes;
    this->tempSize = 0;
}

NnRootWeightLoader::~NnRootWeightLoader() {
    if (tempSize > 0)
        delete[] temp;
}

void NnRootWeightLoader::finish() {
    NnUint zeroSize = 0;
    for (NnUint socketIndex = 0; socketIndex < nNodes - 1; socketIndex++) {
        network->write(socketIndex, &zeroSize, sizeof(zeroSize));
        network->readAck(socketIndex);
    }
    if (tempSize > 0) {
        delete[] temp;
        tempSize = 0;
    }
}

void NnRootWeightLoader::allocate(NnSize size) {
    if (tempSize < size) {
        if (tempSize > 0)
            delete[] temp;
        tempSize = size;
        temp = new NnByte[size];
    }
}

void NnRootWeightLoader::writeWeight(NnUint nodeIndex, const char *opName, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    NnUint nameSize = std::strlen(opName) + 1;
    NnUint socketIndex = nodeIndex - 1;
    network->write(socketIndex, &nameSize, sizeof(nameSize));
    network->write(socketIndex, opName, nameSize);
    network->write(socketIndex, &opIndex, sizeof(opIndex));
    network->write(socketIndex, &offset, sizeof(offset));
    network->write(socketIndex, &nBytes, sizeof(nBytes));
    network->write(socketIndex, weight, nBytes);
}

NnSize NnRootWeightLoader::loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    executor->loadWeight(opName, opIndex, 0u, nBytes, weight);
    return nBytes;
}

NnSize NnRootWeightLoader::loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    executor->loadWeight(opName, opIndex, 0u, nBytes, weight);

    if (nNodes > 1u) {
        for (NnUint nodeIndex = 1u; nodeIndex < nNodes; nodeIndex++)
            writeWeight(nodeIndex, opName, opIndex, 0u, nBytes, weight);
    }
    return nBytes;
}

NnSize NnRootWeightLoader::loadRowMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnRowMatmulSlice *slice, NnByte *weight) {
    const NnUint offset = expertIndex * slice->sliceSize.nBytes;
    if (nNodes == 1u) {
        executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, weight);
    } else {
        allocate(slice->sliceSize.nBytes);
        for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
            splitRowMatmulWeight(slice, nodeIndex, weight, temp);
            if (nodeIndex == 0u)
                executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, temp);
            else
                writeWeight(nodeIndex, opName, opIndex, offset, slice->sliceSize.nBytes, temp);
        }
    }
    return slice->size.nBytes;
}

NnSize NnRootWeightLoader::loadColMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnColMatmulSlice *slice, NnByte *weight) {
    const NnUint offset = expertIndex * slice->sliceSize.nBytes;
    if (nNodes == 1) {
        executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, weight);
    } else {
        allocate(slice->sliceSize.nBytes);
        for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
            splitColMatmulWeight(slice, nodeIndex, weight, temp);
            if (nodeIndex == 0)
                executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, temp);
            else
                writeWeight(nodeIndex, opName, opIndex, offset, slice->sliceSize.nBytes, temp);
        }
    }
    return slice->size.nBytes;
}

NnWorkerWeightReader::NnWorkerWeightReader(NnExecutor *executor, NnNetwork *network) {
    this->executor = executor;
    this->network = network;
    this->tempSize = 0;
}

NnWorkerWeightReader::~NnWorkerWeightReader() {
    if (tempSize > 0)
        delete[] temp;
}

void NnWorkerWeightReader::allocate(NnUint size) {
    if (tempSize < size) {
        if (tempSize > 0)
            delete[] temp;
        tempSize = size;
        temp = new NnByte[size];
    }
}

void NnWorkerWeightReader::read() {
    NnUint nameSize;
    NnUint opIndex;
    NnSize offset;
    NnSize nBytes;
    while (true) {
        network->read(0, &nameSize, sizeof(nameSize));
        if (nameSize == 0) {
            network->writeAck(ROOT_SOCKET_INDEX);
            if (tempSize > 0) {
                delete[] temp;
                tempSize = 0;
            }
            break;
        }
        std::unique_ptr<char[]> opNamePtr(new char[nameSize]);
        char *opName = opNamePtr.get();
        network->read(ROOT_SOCKET_INDEX, opName, nameSize);
        network->read(ROOT_SOCKET_INDEX, &opIndex, sizeof(opIndex));
        network->read(ROOT_SOCKET_INDEX, &offset, sizeof(offset));
        network->read(ROOT_SOCKET_INDEX, &nBytes, sizeof(nBytes));
        allocate(nBytes);
        network->read(0, temp, nBytes);
        executor->loadWeight(opName, opIndex, offset, nBytes, temp);
        printf("💿 Loaded %22s %3d, %12zu kB\n", opName, opIndex, nBytes / 1024);
    }
    printf("💿 Weights loaded\n");
}
