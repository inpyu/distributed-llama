#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SSIZE_T ssize_t;
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "nn/nn-network.hpp"

struct BackendTarget {
    std::string host;
    int port;
    int inflight;
    std::chrono::steady_clock::time_point unhealthyUntil;

    BackendTarget(const std::string &host, int port)
        : host(host), port(port), inflight(0), unhealthyUntil(std::chrono::steady_clock::time_point::min()) {}
};

struct GatewayConfig {
    int listenPort;
    int maxQueue;
    int maxInflightPerBackend;
    int healthRetryMs;
    std::size_t maxBodyBytes;
    std::vector<BackendTarget> backends;

    GatewayConfig()
        : listenPort(9999),
          maxQueue(32),
          maxInflightPerBackend(1),
          healthRetryMs(5000),
          maxBodyBytes(8 * 1024 * 1024) {}
};

struct RuntimeState {
    std::mutex backendsMutex;
    std::size_t rrCursor;
    std::atomic<int> activeRequests;

    RuntimeState() : rrCursor(0), activeRequests(0) {}
};

static void parseBackendAddress(const char *value, std::string *host, int *port) {
    const char *separator = std::strstr(value, ":");
    if (separator == nullptr) {
        throw std::runtime_error("Invalid backend address: " + std::string(value) + " (expected host:port)");
    }

    const std::size_t hostLen = (std::size_t)(separator - value);
    if (hostLen == 0) {
        throw std::runtime_error("Invalid backend address (empty host): " + std::string(value));
    }

    char *end = nullptr;
    const long parsedPort = std::strtol(separator + 1, &end, 10);
    if (end == nullptr || *end != '\0' || parsedPort <= 0 || parsedPort > 65535) {
        throw std::runtime_error("Invalid backend port: " + std::string(separator + 1));
    }

    *host = std::string(value, hostLen);
    *port = (int)parsedPort;
}

static GatewayConfig parseArgs(int argc, char **argv) {
    GatewayConfig config;

    for (int i = 1; i < argc; i++) {
        const char *name = argv[i];
        if (std::strcmp(name, "--listen-port") == 0) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for --listen-port");
            config.listenPort = std::atoi(argv[++i]);
        } else if (std::strcmp(name, "--max-queue") == 0) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for --max-queue");
            config.maxQueue = std::atoi(argv[++i]);
        } else if (std::strcmp(name, "--max-inflight-per-backend") == 0) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for --max-inflight-per-backend");
            config.maxInflightPerBackend = std::atoi(argv[++i]);
        } else if (std::strcmp(name, "--health-retry-ms") == 0) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for --health-retry-ms");
            config.healthRetryMs = std::atoi(argv[++i]);
        } else if (std::strcmp(name, "--max-body-bytes") == 0) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for --max-body-bytes");
            config.maxBodyBytes = (std::size_t)std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(name, "--backends") == 0) {
            i++;
            for (; i < argc && argv[i][0] != '-'; i++) {
                std::string host;
                int port = 0;
                parseBackendAddress(argv[i], &host, &port);
                config.backends.push_back(BackendTarget(host, port));
            }
            i--;
        } else {
            throw std::runtime_error("Unknown option: " + std::string(name));
        }
    }

    if (config.listenPort <= 0 || config.listenPort > 65535) {
        throw std::runtime_error("listen port must be in range 1..65535");
    }
    if (config.maxQueue < 1) {
        throw std::runtime_error("max queue must be >= 1");
    }
    if (config.maxInflightPerBackend < 1) {
        throw std::runtime_error("max inflight per backend must be >= 1");
    }
    if (config.healthRetryMs < 100) {
        throw std::runtime_error("health retry ms must be >= 100");
    }
    if (config.maxBodyBytes < 1024) {
        throw std::runtime_error("max body bytes must be >= 1024");
    }
    if (config.backends.empty()) {
        throw std::runtime_error("At least one backend is required. Example: --backends 10.0.0.1:9001 10.0.0.5:9002");
    }

    return config;
}

static std::size_t findHeaderEnd(const std::string &headerData) {
    const std::string delimiter = "\r\n\r\n";
    return headerData.find(delimiter);
}

static std::size_t parseContentLength(const std::string &headerData) {
    std::size_t contentLength = 0;
    std::size_t lineStart = 0;

    while (lineStart < headerData.size()) {
        const std::size_t lineEnd = headerData.find("\r\n", lineStart);
        if (lineEnd == std::string::npos) {
            break;
        }
        if (lineEnd == lineStart) {
            break;
        }

        const std::string line = headerData.substr(lineStart, lineEnd - lineStart);
        const std::size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::transform(key.begin(), key.end(), key.begin(), ::tolower);
            if (key == "content-length") {
                std::string value = line.substr(colon + 1);
                value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
                if (!value.empty()) {
                    contentLength = (std::size_t)std::strtoull(value.c_str(), nullptr, 10);
                }
                break;
            }
        }

        lineStart = lineEnd + 2;
    }

    return contentLength;
}

static std::vector<char> readHttpRequestRaw(int clientFd, std::size_t maxBodyBytes) {
    std::string buffer;
    buffer.reserve(64 * 1024);

    char temp[64 * 1024];
    std::size_t headerEnd = std::string::npos;

    while (headerEnd == std::string::npos) {
        const ssize_t n = recv(clientFd, temp, sizeof(temp), 0);
        if (n <= 0) {
            throw std::runtime_error("Failed reading request headers");
        }
        buffer.append(temp, (std::size_t)n);
        headerEnd = findHeaderEnd(buffer);
        if (buffer.size() > maxBodyBytes + 64 * 1024) {
            throw std::runtime_error("Request too large while reading headers");
        }
    }

    const std::size_t contentLength = parseContentLength(buffer);
    if (contentLength > maxBodyBytes) {
        throw std::runtime_error("Request body exceeds max body bytes");
    }

    const std::size_t bodyStart = headerEnd + 4;
    while (buffer.size() < bodyStart + contentLength) {
        const ssize_t n = recv(clientFd, temp, sizeof(temp), 0);
        if (n <= 0) {
            throw std::runtime_error("Failed reading request body");
        }
        buffer.append(temp, (std::size_t)n);
        if (buffer.size() > bodyStart + contentLength + 16 * 1024) {
            throw std::runtime_error("Received more body bytes than Content-Length");
        }
    }

    return std::vector<char>(buffer.begin(), buffer.begin() + (std::ptrdiff_t)(bodyStart + contentLength));
}

static int connectToBackend(const std::string &host, int port) {
    struct addrinfo hints;
    struct addrinfo *addr = nullptr;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    char portStr[11];
    std::snprintf(portStr, sizeof(portStr), "%d", port);
    const int resolveResult = getaddrinfo(host.c_str(), portStr, &hints, &addr);
    if (resolveResult != 0 || addr == nullptr) {
        throw std::runtime_error("Cannot resolve backend: " + host + ":" + portStr);
    }

    int fd = ::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (fd < 0) {
        freeaddrinfo(addr);
        throw std::runtime_error("Cannot create backend socket");
    }

    const int connectResult = ::connect(fd, addr->ai_addr, addr->ai_addrlen);
    freeaddrinfo(addr);
    if (connectResult != 0) {
        destroySocket(fd);
        throw std::runtime_error("Cannot connect to backend: " + host + ":" + portStr);
    }

    return fd;
}

static void writePlainResponse(int clientFd, int statusCode, const char *statusText, const std::string &body) {
    std::string response;
    response += "HTTP/1.1 ";
    response += std::to_string(statusCode);
    response += " ";
    response += statusText;
    response += "\r\n";
    response += "Content-Type: application/json; charset=utf-8\r\n";
    response += "Connection: close\r\n";
    response += "Content-Length: ";
    response += std::to_string(body.size());
    response += "\r\n\r\n";
    response += body;
    writeSocket(clientFd, response.c_str(), response.size());
}

static int selectBackendAndAcquire(GatewayConfig *config, RuntimeState *state) {
    std::lock_guard<std::mutex> lock(state->backendsMutex);

    const std::size_t n = config->backends.size();
    if (n == 0) {
        return -1;
    }

    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

    int selected = -1;
    int minInflight = std::numeric_limits<int>::max();

    for (std::size_t i = 0; i < n; i++) {
        const std::size_t idx = (state->rrCursor + i) % n;
        BackendTarget &backend = config->backends[idx];
        if (backend.unhealthyUntil > now) {
            continue;
        }
        if (backend.inflight >= config->maxInflightPerBackend) {
            continue;
        }

        if (backend.inflight < minInflight) {
            minInflight = backend.inflight;
            selected = (int)idx;
        }
    }

    if (selected >= 0) {
        config->backends[(std::size_t)selected].inflight++;
        state->rrCursor = ((std::size_t)selected + 1) % n;
    }

    return selected;
}

static void releaseBackend(GatewayConfig *config, RuntimeState *state, int backendIndex, bool markUnhealthy) {
    if (backendIndex < 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(state->backendsMutex);
    BackendTarget &backend = config->backends[(std::size_t)backendIndex];
    if (backend.inflight > 0) {
        backend.inflight--;
    }
    if (markUnhealthy) {
        backend.unhealthyUntil = std::chrono::steady_clock::now() + std::chrono::milliseconds(config->healthRetryMs);
    }
}

static void proxyResponse(int backendFd, int clientFd) {
    char buf[16 * 1024];
    while (true) {
        const ssize_t n = recv(backendFd, buf, sizeof(buf), 0);
        if (n == 0) {
            break;
        }
        if (n < 0) {
            throw std::runtime_error("Error while reading backend response");
        }
        writeSocket(clientFd, buf, (std::size_t)n);
    }
}

static void processClient(int clientFd, GatewayConfig *config, RuntimeState *state) {
    NnSocket clientSocket(clientFd);
    const int activeNow = ++state->activeRequests;
    if (activeNow > config->maxQueue) {
        --state->activeRequests;
        writePlainResponse(clientSocket.fd, 429, "Too Many Requests", "{\"error\":\"gateway queue is full\"}");
        return;
    }

    int backendIndex = -1;
    bool backendFailed = false;

    try {
        std::vector<char> requestRaw = readHttpRequestRaw(clientSocket.fd, config->maxBodyBytes);
        backendIndex = selectBackendAndAcquire(config, state);
        if (backendIndex < 0) {
            --state->activeRequests;
            writePlainResponse(clientSocket.fd, 429, "Too Many Requests", "{\"error\":\"all replicas are busy or unhealthy\"}");
            return;
        }

        BackendTarget backendCopy("", 0);
        {
            std::lock_guard<std::mutex> lock(state->backendsMutex);
            backendCopy = config->backends[(std::size_t)backendIndex];
        }

        NnSocket backendSocket(connectToBackend(backendCopy.host, backendCopy.port));
        writeSocket(backendSocket.fd, requestRaw.data(), requestRaw.size());
        proxyResponse(backendSocket.fd, clientSocket.fd);
    } catch (const std::exception &e) {
        backendFailed = true;
        try {
            writePlainResponse(clientSocket.fd, 502, "Bad Gateway", "{\"error\":\"backend forwarding failed\"}");
        } catch (...) {
        }
        printf("🚨 Gateway error: %s\n", e.what());
    }

    releaseBackend(config, state, backendIndex, backendFailed);
    --state->activeRequests;
}

int main(int argc, char **argv) {
#ifndef _WIN32
    std::signal(SIGPIPE, SIG_IGN);
#endif
    initSockets();

    int returnCode = EXIT_SUCCESS;
    try {
        GatewayConfig config = parseArgs(argc, argv);
        RuntimeState state;

        printf("🚁 Gateway listen port: %d\n", config.listenPort);
        printf("🚁 Backends: %zu, maxQueue=%d, maxInflightPerBackend=%d\n",
               config.backends.size(),
               config.maxQueue,
               config.maxInflightPerBackend);

        NnSocket serverSocket(createServerSocket(config.listenPort));
        while (true) {
            int clientFd = acceptSocket(serverSocket.fd);
            std::thread worker(processClient, clientFd, &config, &state);
            worker.detach();
        }
    } catch (const std::exception &e) {
        printf("🚨 Critical error: %s\n", e.what());
        returnCode = EXIT_FAILURE;
    }

    cleanupSockets();
    return returnCode;
}
