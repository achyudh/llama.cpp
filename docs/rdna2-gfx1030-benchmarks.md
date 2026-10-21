# RDNA2 (gfx1030) adaptive KV streaming: 40K–128K experimental benchmark (2026-10-20)

## Hardware and settings
- Debian 13; AMD RX 6950 XT 16 GiB, gfx1030, ROCm 10.2 Preview
- Source: achyudh/llama.cpp adaptive KV streamer plus downstream RDNA2 paged attention and independent SDMA engine fixes. Validated on PourrezJ/llama.cpp rdna2/gfx1030 at 2a98649 before this report.
- Model Swift 1.5 Qwen3.8 27B IQ3_S + external MTP2 Q4_0
- Context configured 131072; stream KV K=q8_0 V=q4_0; Flash Attention; GPU ROCm1
- KV resident 256 MiB, one slot, batch/ubatch 256, MTP2 N-max=2, fixed 96 output tokens
- 96 generated tokens are forced with ignore_eos=true, deterministic temperature=0 and cache_prompt=true
- Synthetic needle PEARL-31726 near the beginning of prompts (not a general reasoning quality certification)

## Reproduced measurements

| Prompt tokens | New prompt tokens evaluated | Prefill tok/s | Decode tok/s | Draft accepted / drafted | Marker correct |
|---:|---:|---:|---:|---:|---|
| 39,971 | 39,971 | 368.48 | 26.26 | 57 / 74 | yes |
| 111,970 | 72,255 | 213.50 | 13.74 | 58 / 73 | yes |
| 127,871 | 16,157 | 160.46 | 6.43 | 55 / 78 | yes |

The 112K and 128K requests reuse the previous prompt prefix. Prefill throughput applies only to newly evaluated suffixes. Decode throughput covers 96 generated tokens and reflects MTP2 acceptance; this is not a single-token latency benchmark.
The 127871-token prompt completed and the final generated length was 96, comfortably below 131072.
No observed GPU crash, HIP errors, or OOM in the isolated test. GPU VRAM at 128K prefill ~13.9 GiB.
RAM was pressured and system swap used ~10-14 GiB during part of the experiment, which may contribute to variance.

## Findings

1. The missing short-batch page-run propagation bug was the major source of the severe regression. Clean fork before fix at 112K: 0.746 tok/s; after fix: 13.737 tok/s (~18.4x).
2. Corrected fork 40K decode: 26.26 tok/s; 112K: 13.74 tok/s; 128K: 6.43 tok/s.
3. The 128K target of 20–25 tok/s remains **unmet**. A performance inflection occurs toward full context; root cause still unprofiled.
4. The draft acceptance at 128K is 55/78 (~70.5%) versus 58/73 (~79.5%) at 112K; acceptance accounts for some, not conclusively all, of the drop.
5. The server reported 109 graphs reused / 468 graph reallocs at 128K, versus other runs; investigate graph scheduling, memory pressure, and per-token KV I/O.
6. Results are for Q8_0/Q4_0 main KV. Earlier 112K reference of ~14–15 tok/s used Q4_0/Q4_0 main KV. That is not an exact apples-to-apples comparison.

## Runtime and safety

- Experimental server was isolated; the usual inference backend was restored after measurements.
- No hangs, GPU resets, HIP errors, or OOM were observed in this run; this does not replace validation on other hardware.
- This evidence does not establish correctness for arbitrary 128K workloads or performance parity with RDNA4.

## Reproduction notes / evidence

- Test runner used an isolated llama-server, fixed 131072 context and one slot. Model offloaded fully to ROCm1; production instance was stopped only while testing.
- Requests to /completion were sequential, with cache_prompt=true, temperature=0, n_predict=96, ignore_eos=true, and a repeated prefix across 40K/112K/128K.
- The original local experiment retains full server logs and scripts; these are not bundled in this PR.
- Raw local measurements are preserved for each prompt size (40K, 112K, 128K). The synthetic marker retrieval is a narrow smoke test, not a multi-needle benchmark.
- Main stream flags: --kv-stream --kv-resident-mib 256 -ctk q8_0 -ctv q4_0 --flash-attn on; external draft enabled with --spec-type draft-mtp --spec-draft-n-max 2.

## Next experiments

- Equalize disk pressure and system RAM swap, profile HSA SDMA throughput + per-layer attention time at 112K/128K.
- Quantify how much of speed reduction comes from MTP2 (compare no MTP at 128K after one cached prefill), versus main KV q8/q4 attention.
- Test effective quality with dispersed multi-needle / instruction-following queries, not only the single marker.
- Do not treat this 128K run as a 20–25 token/s performance qualification; further profiling and quality tests are required.
