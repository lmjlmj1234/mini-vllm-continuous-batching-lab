# mini-vLLM Serving Layer Sequence Diagrams

## Scenario 1: Normal Request Completion

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI as FastAPI / ServingLayer
    participant Engine as LLMEngine / EngineCore
    participant Scheduler as Scheduler
    participant BlockMgr as BlockManager
    participant PrefixCache as PrefixCache
    participant Metrics as MetricsCollector

    Client->>FastAPI: POST /generate(prompt, max_tokens)

    Note over FastAPI: --- Admission Phase ---
    FastAPI->>FastAPI: tokenize(prompt)
    FastAPI->>FastAPI: AdmissionControl.check()
    FastAPI->>FastAPI: RateLimiter.check()

    FastAPI->>Engine: add_request(prompt, max_new_tokens)
    Engine->>Engine: create SequenceGroup
    Engine->>Engine: add to RequestQueue._waiting
    Engine-->>FastAPI: engine_rid

    FastAPI->>Engine: run_until_done()

    Note over Engine,Metrics: --- Engine Step Loop (repeats until done) ---

    loop each step() until num_waiting=0 and num_running=0

        Engine->>Engine: _check_timeouts() [no-op for normal flow]

        Engine->>Scheduler: schedule()

        Note over Scheduler: Phase 1: Finish Check
        Scheduler->>BlockMgr: free(seq_id) [for completed sequences]
        BlockMgr->>BlockMgr: pop BlockTable, get physical block IDs
        BlockMgr->>BlockMgr: BlockAllocator.free(pids)
        BlockMgr-->>Scheduler: done

        Note over Scheduler: Phase 2-4: Categorize & Budget

        Note over Scheduler: Phase 5: Admit New Groups (first step only)
        Scheduler->>BlockMgr: probe_prefix_cache(prompt_token_ids)
        BlockMgr->>PrefixCache: lookup(hash)
        PrefixCache-->>BlockMgr: match result (miss if cold)
        BlockMgr-->>Scheduler: PrefixCacheProbeResult (cached_token_count=0)

        Scheduler->>BlockMgr: allocate_for_seq(seq)
        BlockMgr->>BlockMgr: create BlockTable
        BlockMgr-->>Scheduler: shared blocks

        Scheduler-->>Engine: ScheduleResult

        Note over Engine: --- Prefill Phase (first steps) ---
        Engine->>Engine: executor.prefill(scheduled_seqs)

        loop each prefill position
            Engine->>BlockMgr: ensure_block(seq, position)
            BlockMgr->>PrefixCache: lookup(hash)
            alt Cache Hit and ref_count > 0
                PrefixCache-->>BlockMgr: physical_block_id
                BlockMgr->>BlockMgr: increment_ref(pid)
                BlockMgr->>BlockMgr: table.add_shared_block(pid)
            else Cache Miss
                BlockMgr->>BlockMgr: BlockAllocator.allocate(1)
                BlockMgr->>PrefixCache: insert(hash, pid)
            end
            BlockMgr-->>Engine: physical block ID
        end

        Engine->>Engine: write KV cache for prompt tokens
        Engine->>Engine: first_token_time = now, status=RUNNING

        Note over Engine: --- Decode Phase (subsequent steps) ---
        Engine->>Engine: executor.decode(scheduled_seqs)
        Engine->>Engine: sample next token
        Engine->>BlockMgr: ensure_block(seq, pos) [for new KV]
        Engine->>Engine: append output_token_ids

        Note over Engine: --- End-of-Step ---
        Engine->>Engine: cleanup finished sequences
        Engine->>Metrics: register_sequence(seq)
        Engine->>Metrics: record_step(result, sched_latency, step_wall, total_blocks, used_blocks)
    end

    Note over Engine: Last step: Scheduler Phase 1 detects num_generated >= max_tokens
    Engine->>BlockMgr: free(seq_id) [final release]
    BlockMgr->>PrefixCache: (shared blocks preserved, ref counts decremented)

    Engine-->>FastAPI: {engine_rid: output_text}

    FastAPI->>FastAPI: RateLimiter.record(estimated)
    FastAPI->>FastAPI: StreamManager.release()
    FastAPI-->>Client: ServingResponse(text=output_text)
```

---

## Scenario 2: Client Disconnect

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI as FastAPI / ServingLayer
    participant CancelMgr as CancelManager
    participant Engine as LLMEngine / EngineCore
    participant BlockMgr as BlockManager
    participant Metrics as MetricsCollector

    Note over Client: Client closes connection mid-generation
    Client-->>FastAPI: TCP connection closed / HTTP disconnect

    Note over FastAPI: ⚠️ Limitation: No automatic disconnect detection
    Note over FastAPI: The synchronous engine loop cannot be interrupted.
    Note over FastAPI: The intended mechanism is explicit cancellation:

    FastAPI->>CancelMgr: cancel(request_id)

    CancelMgr->>Engine: cancel_request(request_id)

    Engine->>Engine: lookup SequenceGroup by request_id

    loop each unfinished sequence in group
        Engine->>Engine: seq.status = CANCELLED
        Engine->>Engine: seq.finish_time = now

        Engine->>BlockMgr: free(seq.seq_id)
        BlockMgr->>BlockMgr: pop BlockTable
        BlockMgr->>BlockMgr: get physical block IDs
        BlockMgr->>BlockMgr: BlockAllocator.free(pids)
        Note over BlockMgr: ref_count decremented; block returns to free pool only when ref_count == 0

        Engine->>Engine: executor.cleanup_sequence(seq_id)
        Engine->>Metrics: register_sequence(seq)
    end

    Engine->>Metrics: count_cancelled()

    Engine->>Engine: remove from _running / _waiting
    Engine->>Engine: move to _finished
    Engine-->>CancelMgr: done

    CancelMgr-->>FastAPI: success

    FastAPI-->>Client: (no response — connection already closed)
```

---

## Scenario 3: Request Timeout

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI as FastAPI / ServingLayer
    participant Engine as LLMEngine / EngineCore
    participant Scheduler as Scheduler
    participant BlockMgr as BlockManager
    participant Metrics as MetricsCollector

    Client->>FastAPI: POST /generate(prompt, max_tokens)
    FastAPI->>Engine: add_request(prompt, max_new_tokens)
    FastAPI->>Engine: run_until_done()

    Note over Engine: Config.request_timeout_s = 60.0 (default)

    loop each step()
        Note over Engine,Rightside: ⏱️ _check_timeouts() runs at the TOP of EVERY step

        Engine->>Engine: _check_timeouts()
        Engine->>Engine: now = time.time()

        alt running group: now - arrival_time > timeout
            Engine->>Engine: timed_out_sgs.append(sg)
        else waiting group: now - arrival_time > timeout
            Engine->>Engine: timed_out_sgs.append(sg)
        end

        Note over Engine: For each timed-out group:

        Engine->>Engine: seq.status = TIMEOUT
        Engine->>Engine: seq.finish_time = now

        Engine->>BlockMgr: free(seq.seq_id)
        BlockMgr->>BlockMgr: pop BlockTable
        BlockMgr->>BlockMgr: BlockAllocator.free(pids)
        Note over BlockMgr: Shared prefix blocks: only decrement ref_count
        Note over BlockMgr: Block returns to free pool only when ref_count == 0

        Engine->>Engine: executor.cleanup_sequence(seq_id)
        Engine->>Metrics: register_sequence(seq)
        Engine->>Metrics: count_timeout()

        Engine->>Engine: remove from _running / _waiting
        Engine->>Engine: move to _finished

        Note over Engine: After timeout processing, the step continues
        Note over Engine: for remaining (non-timed-out) sequences...

        Engine->>Scheduler: schedule() [for remaining sequences]
        Scheduler-->>Engine: ScheduleResult

        Engine->>Engine: executor.prefill/se_decode
        Engine->>Metrics: record_step(...)
    end

    Note over Engine: Engine loop sees num_waiting=0, exits
    Engine-->>FastAPI: partial or empty output (timed-out requests not in outputs)

    FastAPI-->>Client: ServingResponse(error="timeout", partial_text="...")
```

---

## Component Reference: Step Lifecycle (All Three Flows Overlaid)

```mermaid
flowchart TD
    A["EngineCore.step() start"] --> B["_check_timeouts()"]

    B --> C{"Timed-out groups?"}
    C -->|Yes| D["For each: free() + cleanup() + count_timeout()"]
    D --> E
    C -->|No| E["Scheduler.schedule()"]

    E --> F["Phase 1: Finish Check"]
    F --> F1["free() finished seq"]
    F1 --> G["Phase 2-4: Categorize + Budget"]

    G --> H["Phase 5: Admit New"]
    H --> H1["probe_prefix_cache()"]
    H --> H2["allocate_for_seq()"]

    H1 --> H3["PrefixCache.lookup()"]
    H2 --> H4["BlockTable created"]

    H --> I["Phase 6: Token Counts"]

    I --> J["executor.prefill()"]
    J --> J1["ensure_block() → PrefixCache lookup/insert"]
    J1 --> K["executor.decode()"]

    K --> L["Cleanup finished sequences"]
    L --> L1["register_sequence()"]
    L --> L2["record_step()"]

    L2 --> M["EngineCore.step() end"]
```

---

## Notes

1. **`BlockManager.free()` 是级联释放**：BlockManager 释放 BlockTable 时，会获得所有物理块 ID，逐一调用 `BlockAllocator.free(pid)`。后者的 `ref_count` 递减，只有当 `ref_count == 0` 时，物理块才真正归还空闲池。这使得通过 PrefixCache 共享的块在其中一个使用者释放后不会归还。

2. **`ensure_block()` 的双重 PrefixCache 检查**：Scheduler 的 `probe_prefix_cache()` 是只读探针（不改变引用计数），而 Executor 的 `ensure_block()` 在写入 KV 时会再次查询 PrefixCache。这是因为两次调用之间有间隔，另一个序列可能在中间注册了共享块。

3. **Timeout vs Cancel**：Timeout 在 EngineCore 内部自动触发（`_check_timeouts()` 在每个 step 顶部运行），而 Cancel 由 Serving 层通过 CancelManager 外部触发。但二者最终调用相同的释放路径（`free()` + `cleanup_sequence()` + `register_sequence()`），只是 metrics 计数器不同（`count_timeout()` vs `count_cancelled()`）。

4. **Client Disconnect 未自动检测**：当前架构是同步阻塞的，Engine 在 `run_until_done()` / `step()` 内部运行时无法被中断。用户需通过 `/cancel` 端点手动取消。
