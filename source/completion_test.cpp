#include <iostream>
#include <print>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    std::string host = "100.94.72.0";
    int port = 4000;
    std::string body = "{\"model\": \"llama_cpp/generic\", \"prompt\":\"write a poem about cherry blossoms\", \"stream\":true}";

    std::string request =
        "POST /v1/completions HTTP/1.1\r\n"
        "Host: " + host + ":4000\r\n"
        "Authorization: Bearer sk-wl7EXdnD_uDYsAaDtxtqZA\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body.length()) + "\r\n"
        "Connection: close\r\n"
        "\r\n" +
        body;

    std::println("[LOG] Starting client...");

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    bool socket_failed = (sock < 0);
    if (socket_failed) {
        perror("[ERROR] Socket creation failed");
        return 1;
    }
    std::println("[LOG] Socket created successfully.");

    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    bool invalid_address = (inet_pton(AF_INET, host.c_str(), &serv_addr.sin_addr) <= 0);
    if (invalid_address) {
        perror("[ERROR] Invalid address");
        return 1;
    }
    std::println("[LOG] Address parsed: {}", host);

    std::println("[LOG] Attempting to connect to {}:{}...", host, port);
    bool connect_failed = (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0);
    if (connect_failed) {
        perror("[ERROR] Connection failed");
        return 1;
    }
    std::println("[LOG] Connected successfully.");

    std::println("[LOG] Sending request...");
    bool send_failed = (send(sock, request.c_str(), request.length(), 0) < 0);
    if (send_failed) {
        perror("[ERROR] Send failed");
        return 1;
    }
    std::println("[LOG] Request sent. Waiting for response...");

    char buffer[4096];
    int bytes_received;
    std::string stream_buffer;
    size_t header_end;
    size_t data_pos;

    while (true) {
        bytes_received = recv(sock, buffer, sizeof(buffer) - 1, 0);
        bool received_data = (bytes_received > 0);
        if (!received_data) {
            break;
        }
        buffer[bytes_received] = '\0';
        stream_buffer += std::string(buffer);

        header_end = stream_buffer.find("\r\n\r\n");
        bool has_header_end = (header_end != std::string::npos);
        if (has_header_end) {
            stream_buffer.erase(0, header_end + 4);
        }

        data_pos = stream_buffer.find("data: ");
        bool has_data_line = (data_pos != std::string::npos);
        while (has_data_line) {
            size_t line_end = stream_buffer.find("\n", data_pos);
            bool missing_line_end = (line_end == std::string::npos);
            if (missing_line_end) {
                break;
            }

            std::string line = stream_buffer.substr(data_pos + 6, line_end - (data_pos + 6));
            bool ends_with_cr = (!line.empty() && line.back() == '\r');
            if (ends_with_cr) {
                line.pop_back();
            }

            bool is_done = (line == "[DONE]");
            if (is_done) {
                std::println();
                goto end_loop;
            }

            try {
                auto j = json::parse(line);
                std::println(stderr, "[DEBUG] Parsed JSON: {}", j.dump());
                bool has_choices = (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty());
                if (has_choices) {
                    bool has_text = (j["choices"][0].contains("text") && j["choices"][0]["text"].is_string());
                    if (has_text) {
                        std::string text = j["choices"][0]["text"];
                        std::println(stderr, "[DEBUG] Queried path: choices[0].text");
                        std::print("{}", text);
                        std::fflush(stdout);
                    }
                }
            } catch (const json::parse_error& e) {
                std::println(stderr, "[DEBUG] Parse error: {}", e.what());
                break;
            }
            stream_buffer.erase(0, line_end + 1);

            data_pos = stream_buffer.find("data: ");
            has_data_line = (data_pos != std::string::npos);
        }
    }

end_loop:
    bool connection_closed = (bytes_received == 0);
    bool receive_failed = (bytes_received < 0);
    if (connection_closed) {
        std::println("[LOG] Connection closed by server.");
    } else if (receive_failed) {
        perror("[ERROR] Receive failed");
    }

    close(sock);
    std::println("[LOG] Client finished.");
    return 0;
}
