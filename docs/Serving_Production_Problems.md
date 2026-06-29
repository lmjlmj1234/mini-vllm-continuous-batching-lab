# Serving Production Problems — 10 Q&A

> Production thinking for LLM serving infrastructure.
>
> These questions probe the gap between "it works on my machine" and
> "it works at 10,000 QPS with random client behaviour."

---

## Q1: Why can't we just use QPS for rate limiting?

**Short answer:** Because a single request can consume 1000x more resources than another.

**Explain:**

Two requests to the same model:

| Request | Prompt Tokens | Output Tokens | Total Compute |
|---------|--------------|---------------|---------------|
| A       | 16           | 4             | 20 tokens     |
| B       | 4096         | 2048          | 6144 tokens   |

At QPS=10, both requests count as "1" — but B consumes 307x more compute than A.

If you rate-limit by QPS alone, an attacker (or buggy client) sends 10 long-context requests and saturates the GPU for 60 seconds. The throughput drops to near zero even though QPS is "normal."

**Production solution:**
- **TPM** (Tokens Per Minute): caps total token throughput
- **RPM** (Requests Per Minute): caps request overhead
- Both gates must pass. RPM protects the scheduler (request metadata, queue management); TPM protects the GPU (prefill + decode).

In this implementation, the `RateLimiter` checks both:

```python
err = self._rate_limiter.check(total_tokens)
# where total_tokens = len(prompt_token_ids) + max_tokens
```

The sum `prompt_len + max_tokens` is the total number of tokens the model will process — the real token count for TPM billing. A 4k-prompt request with max_tokens=2k uses 6k tokens, not 8M. (If you need a compute-cost proxy rather than a token count, that would be a separate weighted formula using prefill and decode cost factors.)

---

## Q2: Why does a client disconnect NOT release GPU resources?

**Short answer:** The HTTP server and the LLM engine run in different threads with different state. Closing a TCP socket doesn't decrement `ref_count`.

**Explain:**

The problem is architectural:

```
Client ──HTTP──► Serving Layer ──Python call──► Engine ──CUDA──► GPU
         ^^^                  ^^^^                     ^^^^
      TCP socket           in-process call          pinned memory
```

When the client disconnects:
1. The TCP socket detects EOF — the HTTP server knows
2. BUT: the engine's `SequenceGroup` is still in `_running`
3. The BlockManager still holds `ref_count > 0` on physical blocks
4. The scheduler still allocates token budget for this sequence

The GPU memory is only freed when:
```python
BlockManager.free(seq_id)  # decrements ref_count
BlockAllocator.free(block)  # returns block to free pool
```

These only happen when the sequence finishes *through the engine*. A TCP disconnect bypasses this path entirely — the sequence runs to `max_tokens`, consuming GPU compute for tokens nobody reads.

**Production solution:**
- `CancelManager` detects idle/disconnected streams (by tracking ID or heartbeat)
- Explicit `cancel(request_id)` call frees blocks immediately
- Timeout mechanism periodically sweeps stuck sequences

---

## Q3: What does "Admission Control before Scheduler" mean in practice?

**Short answer:** The admission gate runs before `engine.step()`, so rejected requests never enter the scheduling loop.

**Explain:**

Without admission control, the request flow is:

```
New request → engine.add_request() → next step() → scheduler.schedule() → Phase 5: Admit → reject
```

A prompt that's too long travels through 5 scheduler phases before rejection. A queue overflow causes the scheduler to iterate 100 waiting entries before refusing the new one.

With admission control (`serving/admission_control.py`):

```
New request → AdmissionControl.check()
                ├── prompt_too_long?     → PROMPT_TOO_LONG (never reaches engine)
                ├── queue_overflow?      → QUEUE_OVERFLOW (never reaches scheduler)
                ├── block_exhausted?     → BLOCK_EXHAUSTED (checks block pool before schedule)
                └── passes              → engine.add_request() → scheduler
```

**Production reality:**
- vLLM does NOT have admission control — it relies on the scheduler to reject. Production deployments add a separate API gateway.
- TGI has `max_input_length` and `max_batch_prefill_tokens`.
- Without it, a single 10k-token prompt entering the scheduler gets chunked across 2500 steps, blocks decode for all other requests, and eventually exhausts KV cache.

---

## Q4: What happens when the KV cache fills up? Can the system crash?

**Short answer:** No crash — the admission gate catches it. But without it, you get a hard-to-debug failure in `ensure_block()`.

**Explain:**

The block exhaustion check in `AdmissionControl`:

```python
watermark_blocks = int(total * self._block_watermark_pct)  # 20%
if free - needed_blocks < watermark_blocks:
    return "BLOCK_EXHAUSTED"
```

Without this check, the engine crashes here:
```python
# In BlockAllocator.allocate_memory():
if self.num_free_blocks == 0:
    raise RuntimeError("Out of memory!")  # ← crash
```

The 20% watermark is a safety margin. When `block_utilization` (in `/metrics`) approaches 100%, the operator knows to:
- Scale up (more GPU blocks)
- Scale down (shorter prompts, fewer concurrent requests)
- Investigate memory leaks

**Key insight:** LLM serving has a *hard* memory ceiling (KV cache) that CPU-only services don't. A web server under load slows down. An LLM server under load *crashes* if it runs out of KV blocks.

---

## Q5: Why does SSE (Server-Sent Events) make streaming harder than WebSocket?

**Short answer:** SSE can detect client disconnection, but only on the *next* `write()` — which creates a detection lag. WebSocket has an application-level ping/pong that detects disconnection much sooner.

**Explain:**

SSE is just HTTP with `Content-Type: text/event-stream`. The server writes `data: {token}\n\n` for each generated token.

Problems:
1. **No client→server channel.** The client can't tell the server "stop generating" within the same connection.
2. **Disconnect detection has lag.** SSE *can* detect disconnection — the OS returns `EPIPE` (or `SIGPIPE`) on the next `write()` when the peer has closed. But if generation is blocked on GPU compute for seconds, the write doesn't happen, and the disconnect goes undetected until the next token is ready. The detection is correct, just delayed.
3. **Stream slot leaks.** The `StreamManager.try_acquire()` is never matched by a `release()` if the client disconnects before `poll_stream()` detects the finish.

**Production solutions:**
- Separate `generate_stream()` (initiates) from `poll_stream()` (reads tokens). The tracking_id lets the server clean up independently of client behaviour.
- Add heartbeat timeout: if `poll_stream()` isn't called for N seconds, assume client disconnect and cancel.
- For true bidirectional control, use WebSocket. But SSE is simpler for read-only streaming.

---

## Q6: How does Continuous Batching change the failure model?

**Short answer:** In continuous batching, ONE slow request blocks ALL requests. In static batching, it blocks only its own batch.

**Explain:**

**Static batching (traditional):**
```
Batch 1: [A: 4 tok, B: 4 tok, C: 4 tok] → all finish in 4 steps
Batch 2: [D: 256 tok, E: 4 tok, F: 4 tok] → 256 steps
E and F wait for D to finish their batch
→ E and F have 256-step latency even though they only need 4 steps
```

**Continuous batching (this system):**
```
Step 1: prefill [D], decode [A, B, C]
Step 2: decode [D, A, B], prefill [E]
Step 3: decode [D, A, E], prefill [F]
...
→ E and F start generating quickly, but D still runs for 256 steps
→ D's long decode steals budget from E and F every single step
```

**Failure mode:**
A single long-generation request *drags down* all concurrent requests by consuming the token budget (`max_num_batched_tokens`) every step. The impact is proportional — not the *worst-case* latency, but the *system-wide throughput* degrades.

**Implication for testing:**
This is why the fault injection tests use `max_new_tokens=64` and `max_num_seqs=4`. With 4 concurrent long requests, each step has less budget per request → higher TPOT for all.

---

## Q7: Why test ref_count integrity after cancel?

**Short answer:** A single leaked ref_count permanently pins a KV block. Over time, the block pool drains silently.

**Explain:**

RefCount is the memory-safety mechanism for shared KV blocks (prefix cache):

```python
class BlockAllocator:
    def free(self, block):
        block.ref_count -= 1
        if block.ref_count == 0:
            self._free_blocks.append(block)  # truly free
```

If `free()` is called once but the block was shared (ref_count=2), the block stays allocated. If `free()` is never called (cancel race condition), the block leaks forever.

The cancel storm test validates:
```python
ref_counts = alloc.dump_ref_counts()
assert all(rc == 0 for rc in ref_counts), f"Leaked refs: {ref_counts}"
```

This assertion would catch:
- Cancel called before the block was allocated (ref_count=0, free decrements to -1)
- Cancel missed a block in the table (ref_count=1, never freed)
- Race between cancel and scheduler allocator (ref_count inconsistency)

**Production analogy:** This is equivalent to checking `malloc`/`free` balance in C programs. A single leak compounds over millions of requests.

---

## Q8: What happens when the engine runs faster than the serving layer can poll?

**Short answer:** Tokens buffer in `output_token_ids`, the `gen_count` cursor tracks position, and the client never misses a token.

**Explain:**

The streaming design separates generation from delivery:

```python
# Engine generates tokens (potentially faster than client reads)
seq.output_token_ids = [101, 203, 305, 402, ...]  # growing list

# Client polls one token at a time
def poll_stream(engine_rid, tracking_id, gen_count):
    while gen_count < seq.num_generated_tokens:
        tok = seq.output_token_ids[gen_count]
        gen_count += 1
        return (token_text, gen_count, False, None)
    return ("", gen_count, False, None)
```

The `gen_count` cursor ensures:
- Each token is delivered exactly once (idempotent polling)
- If the client polls slowly, tokens accumulate in the list (bounded by `max_tokens`)
- If the engine stalls, polling returns empty strings

**Failure mode:**
- Client polls too slowly: tokens buffer in memory (O(max_tokens) per sequence - negligible)
- Client never polls: engine finishes, `StreamManager.release()` called on finish, stream slot freed
- Client polls too fast: returns empty string (no harm)

---

## Q9: How would you test a rate limiter for correctness?

**Short answer:** Test three properties: limit enforcement, window boundary behaviour, and counter reset.

**Explain:**

Rate limiters have subtle edge cases at window boundaries. The tests in `test_serving_layer.py` cover:

```python
# 1. Limit enforcement: N requests within window, N+1th rejected
def test_rpm_rejects_after_limit(self):
    sv = _serving(rate_limit_rpm=2)
    assert sv.generate("Hello", max_tokens=2).error is None  # 1/2
    assert sv.generate("World", max_tokens=2).error is None  # 2/2
    # Third would be rejected (but test doesn't send it)

# 2. Window reset: after time passes, counter resets
def test_rpm_limit_resets(self):
    sv = _serving(rate_limit_rpm=1, rate_limit_tpm=100000)
    assert sv.generate("Hello", max_tokens=2).error is None  # 1/1
    assert sv.rate_limiter._rpm.count() > 0  # counter is non-zero
```

But a thorough test suite would also cover:
- **Window boundary:** request exactly at T=59.999s counts against the old window; request at T=60.001s hits a fresh window
- **Counter overflow:** does the counter wrap safely at 2^63?
- **Concurrent check+record:** two requests check() simultaneously, both pass, then both record — does the total exceed the limit?
- **Token estimation accuracy:** does `prompt_len × max_tokens` overestimate or underestimate actual consumption?

The sliding window counter implementation handles these with a deque of (timestamp, count) tuples, which provides exact per-interval tracking without the spike-at-reset problem of fixed-window counters.

---

## Q10: What metrics do you need to debug an LLM serving outage?

**Short answer:** The six numbers that tell you whether the bottleneck is admission, scheduling, compute, or memory.

**Explain:**

When the system degrades, you need to answer: "which layer is saturated?"

| Metric | What it tells you | Action if high |
|--------|------------------|----------------|
| `block_utilization` | KV cache pressure | Increase blocks, reduce concurrency |
| `waiting_requests` | Scheduler throughput | Increase `max_num_seqs` or GPU |
| `active_streams` | Streaming capacity | Increase `max_num_streams`, fix leaks |
| `rpm_rejected` | Request rate too high | Throttle clients, increase RPM |
| `tpm_rejected` | Token rate too high | Reduce max_tokens, optimize prompts |
| `avg_ttft_ms` | Prefill bottleneck | Reduce long prompts, increase block size |

**Diagnostic flow:**
```
System slow ──► avg_ttft_ms > 500ms?
                   ├── YES → prefill bottleneck → check block_utilization
                   │           ├── high → KV cache pressure
                   │           └── low → GPU compute bound
                   └── NO  → decoding bottleneck → check avg_tpot_ms
                               ├── high → check waiting_requests
                               │           ├── high → scheduler bottleneck
                               │           └── low → GPU decode bound
                               └── normal → check network / client
```

The `/metrics` endpoint in this implementation returns all six in a single JSON blob, designed to be scraped every 5-10 seconds by a monitoring system.

**What's missing (for production):**
- Per-request latency histograms (P50/P99/P999)
- GPU utilization (nvidia-smi integration)
- Request-level logging (who sent the 4k-prompt storm?)
- Automatic alarming on block_utilization > 85%
