#pragma once

#include "ggml-backend.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

struct llama_model;

// Streams the host-resident experts of a MoE model to one device for large batches.
// The expert tensors of a layer form groups, one per matrix multiplication (gate, up, down; each with its EXL3 scales).
// Each group has slots in a device staging buffer: one, or two for the last group of a layer, whose next use follows
// too closely for one. The scheduler's copy of an expert tensor aliases its slot (no compute buffer, no device copy),
// and a copy thread with its own stream uploads a group into its slot as soon as the slot's previous contents have
// been used: when the scheduler starts the next group, everything that reads the previous one has been queued.
class llama_expert_stream {
public:
    // min_tokens: the smallest batch whose expert copies go through the stream; smaller batches copy what they use
    llama_expert_stream(const llama_model & model, ggml_backend_t backend, int64_t min_tokens);
    ~llama_expert_stream();

    // the staged tensor that the scheduler's copy of w can alias for a batch of n_tokens, or nullptr
    ggml_tensor * alias(ggml_backend_t backend, const ggml_tensor * w, int64_t n_tokens) const;

    // makes dst, the scheduler's copy of the host expert tensor w on backend, hold w for a batch of n_tokens: waits for
    // the upload of w's group, and copies it unless dst aliases the staged tensor
    // false if w is not streamed, the backend differs or the batch is too small
    bool copy(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t n_tokens);

    size_t staging_size() const { return staging_bytes; }

    ggml_backend_t get_backend() const { return backend; }
    ggml_backend_buffer_t staging_buffer() const { return staging; }
    bool streams(int64_t n_tokens) const { return n_tokens >= min_tokens; }
    void set_borrowed(bool value);

private:
    struct slot {
        int                  layer = -1;      // model layer whose group is (being) uploaded
        bool                 queued = false;  // in the upload queue
        bool                 uploaded = false; // upload submitted and recorded in staged
        ggml_backend_event_t staged = nullptr;   // on the copy stream, after the upload
        ggml_backend_event_t consumed = nullptr; // on the compute stream, after everything that read the slot
    };
    struct entry {
        int layer, group, index; // index of the tensor in its layer
    };

    int  slot_of(int il, int group) const;
    int  next_layer(int il, int steps) const;
    void acquire(int il, int group);
    void request(int s, int il); // with the mutex held
    void run();
    void release();

    ggml_backend_t        backend;                  // computes the graphs
    ggml_backend_t        copy_backend = nullptr;   // a second instance of the device, for its own stream
    ggml_backend_buffer_t staging = nullptr;
    ggml_context *        ctx = nullptr;
    size_t                staging_bytes = 0;
    int64_t               min_tokens;

    std::vector<std::vector<const ggml_tensor *>> weights; // [layer] host tensors
    std::vector<std::vector<ggml_tensor *>>       staged;  // [layer] the same tensors in their slots
    std::vector<std::vector<int>>                 group_of; // [layer][tensor]
    std::vector<int>                              order;    // layers with streamed experts
    std::vector<int>                              position; // [layer] index in order, -1 if not streamed
    std::vector<int>                              group_slots, group_first_slot; // [group]
    std::unordered_map<const ggml_tensor *, entry> index;

    // main thread only
    int current_layer = -1, current_group = -1; // the group the compute stream was last made to wait for
    bool borrowed = false;

    // shared with the copy thread
    std::mutex              mutex;
    std::condition_variable cv;
    std::vector<slot>       slots;
    std::deque<int>         queue; // slots to upload, in request order
    bool                    stop = false;
    std::thread             thread;
};

using llama_expert_stream_ptr = std::unique_ptr<llama_expert_stream>;
