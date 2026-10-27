// Decode one ubatch and write the graph tensors it computes, for op-by-op comparisons between backends,
// placements or implementations.
//
// usage: llama-graph-dump [common options: -m, -p/-f, -ngl, -dev, -t, -c, --moe-cache-mib, ...]
//            --dump-out FILE [--dump-tokens FILE] [--dump-n N] [--dump-filter REGEX] [--dump-passes N]
//
// The ubatch is the first N tokens of the prompt (default: all), or the whitespace-separated token ids in
// --dump-tokens, used verbatim. Pass p > 1 clears the memory, decodes the same tokens again and writes FILE.pass<p>.
//
// File format (little endian), version 1:
//   char[8] "LGDUMP\0\1", uint32 version, uint32 pass, uint32 n_tokens, int32 tokens[n_tokens]
//   then one record per contiguous F32 or I32 tensor, in evaluation order:
//   uint32 name_len, char name[name_len], uint32 ggml_type, uint32 ggml_op, int64 ne[4], uint64 n_bytes, data[n_bytes]
// Data is row-major with ne[0] contiguous. Names repeat across graphs; readers match by name and occurrence.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

struct dump_state {
    FILE * out = nullptr;
    bool filter = false;
    std::regex pattern;
    std::vector<uint8_t> buffer;
    size_t records = 0;
};

static bool dump_eligible(const ggml_tensor * t) {
    return (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_I32) && ggml_is_contiguous(t) && !ggml_is_empty(t);
}

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    if (!dump_eligible(t) || (st->filter && !std::regex_search(t->name, st->pattern))) {
        return false;
    }
    if (ask) {
        return true;
    }
    st->buffer.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, st->buffer.data(), 0, st->buffer.size());
    const uint32_t name_len = strlen(t->name), type = t->type, op = t->op;
    const int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    const uint64_t n_bytes = st->buffer.size();
    fwrite(&name_len, 4, 1, st->out);
    fwrite(t->name, 1, name_len, st->out);
    fwrite(&type, 4, 1, st->out);
    fwrite(&op, 4, 1, st->out);
    fwrite(ne, 8, 4, st->out);
    fwrite(&n_bytes, 8, 1, st->out);
    fwrite(st->buffer.data(), 1, n_bytes, st->out);
    st->records++;
    return true;
}

// remove the --dump-* options, which the common parser does not know
static bool take_dump_args(int & argc, char ** argv, std::string & out, std::string & tokens, int & n, std::string & filter, int & passes) {
    int kept = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string * dst = arg == "--dump-out" ? &out : arg == "--dump-tokens" ? &tokens : arg == "--dump-filter" ? &filter : nullptr;
        if (dst || arg == "--dump-n" || arg == "--dump-passes") {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a value\n", arg.c_str());
                return false;
            }
            const char * value = argv[++i];
            if (dst) {
                *dst = value;
            } else {
                (arg == "--dump-n" ? n : passes) = atoi(value);
            }
            continue;
        }
        argv[kept++] = argv[i];
    }
    argc = kept;
    return !out.empty() && passes >= 1;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string out_path, tokens_path, filter;
    int n_dump = 0, passes = 1;
    if (!take_dump_args(argc, argv, out_path, tokens_path, n_dump, filter, passes)) {
        fprintf(stderr, "usage: %s [common options] --dump-out FILE [--dump-tokens FILE] [--dump-n N] [--dump-filter REGEX] [--dump-passes N]\n", argv[0]);
        return 1;
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    dump_state st;
    if (!filter.empty()) {
        st.filter = true;
        st.pattern = std::regex(filter);
    }
    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    // the dumped tokens must fit one ubatch of the -b/-ub sizes; checked once they are known
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    std::vector<llama_token> tokens;
    if (!tokens_path.empty()) {
        std::ifstream f(tokens_path);
        for (long long id; f >> id;) {
            tokens.push_back((llama_token) id);
        }
    } else {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    }
    if (n_dump > 0) {
        if ((size_t) n_dump > tokens.size()) {
            LOG_ERR("%s: %d tokens requested, %zu available\n", __func__, n_dump, tokens.size());
            return 1;
        }
        tokens.resize(n_dump);
    }
    if (tokens.empty() || tokens.size() > llama_n_ubatch(ctx) || tokens.size() > llama_n_batch(ctx)) {
        LOG_ERR("%s: %zu tokens must fit one ubatch (n_batch %u, n_ubatch %u)\n", __func__, tokens.size(), llama_n_batch(ctx), llama_n_ubatch(ctx));
        return 1;
    }

    int rc = 0;
    for (int pass = 1; pass <= passes && rc == 0; ++pass) {
        const std::string path = pass == 1 ? out_path : out_path + ".pass" + std::to_string(pass);
        if (pass > 1) {
            llama_memory_clear(llama_get_memory(ctx), true);
        }
        st.out = fopen(path.c_str(), "wb");
        if (!st.out) {
            LOG_ERR("%s: cannot write %s\n", __func__, path.c_str());
            return 1;
        }
        const uint32_t version = 1, upass = pass, n = tokens.size();
        fwrite("LGDUMP\0\1", 1, 8, st.out);
        fwrite(&version, 4, 1, st.out);
        fwrite(&upass, 4, 1, st.out);
        fwrite(&n, 4, 1, st.out);
        fwrite(tokens.data(), sizeof(llama_token), n, st.out);
        st.records = 0;
        rc = llama_decode(ctx, llama_batch_get_one(tokens.data(), n));
        const bool ok = fclose(st.out) == 0;
        st.out = nullptr;
        LOG_INF("%s: pass %d: decode %d, %u tokens, %zu tensors -> %s\n", __func__, pass, rc, n, st.records, path.c_str());
        if (!ok) {
            rc = 1;
        }
    }

    llama_backend_free();
    return rc == 0 ? 0 : 1;
}
