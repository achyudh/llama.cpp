#pragma once

#include "ggml-backend.h"

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

struct llama_model;

// Streams the host-resident experts of a MoE model to one device for large batches, a whole layer at a time.
// A copy thread uploads each layer into a single staging buffer on its own stream while the previous layer computes;
// the scheduler's copy of an expert tensor then becomes a device-to-device copy from the staging buffer. Once the last
// tensor of a layer has been copied out, the thread starts on the next layer (wrapping to the first, for the next
// batch). A layer that was not prefetched is uploaded when first needed.
class llama_expert_stream {
public:
    // min_tokens: the smallest batch whose expert copies go through the stream; smaller batches copy what they use
    llama_expert_stream(const llama_model & model, ggml_backend_t backend, int64_t min_tokens);
    ~llama_expert_stream();

    // copies the host expert tensor w into dst, a tensor of the scheduler on backend, for a batch of n_tokens
    // false if w is not streamed, the backend differs or the batch is too small
    bool copy(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t n_tokens);

    size_t staging_size() const { return staging_bytes; }

private:
    struct layer {
        std::vector<const ggml_tensor *> weights; // host tensors
        std::vector<ggml_tensor *>       staged;  // the same tensors in the staging buffer
    };

    void acquire(int il);
    void request(int il); // with the mutex held
    void run();
    void release();

    ggml_backend_t        backend;        // computes the graphs
    ggml_backend_t        copy_backend = nullptr; // a second instance of the device, for its own stream
    ggml_backend_buffer_t staging = nullptr;
    ggml_backend_event_t  staged_event = nullptr;   // recorded on copy_backend after a layer upload
    ggml_backend_event_t  consumed_event = nullptr; // recorded on backend after the copies out of staging
    ggml_context *        ctx = nullptr;
    size_t                staging_bytes = 0;
    int64_t               min_tokens;

    std::vector<layer>    layers; // by model layer, empty for layers without host experts
    std::unordered_map<const ggml_tensor *, std::pair<int, int>> index; // weight -> layer, position

    // main thread only
    int    current = -1;  // layer whose upload the backend waits for
    size_t n_copied = 0; // tensors of the current layer copied out of staging

    // shared with the copy thread
    std::mutex              mutex;
    std::condition_variable cv;
    int  requested = -1; // layer the thread is to upload, -1 when idle
    int  staged    = -1; // layer whose upload has been submitted and recorded in staged_event
    bool stop = false;
    std::thread thread;
};

using llama_expert_stream_ptr = std::unique_ptr<llama_expert_stream>;
