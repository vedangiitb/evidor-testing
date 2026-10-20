# Evidor Streaming Tokens, Tool Calling & Langfuse Telemetry Testing

This project provides a comprehensive verification test suite for testing the newly introduced **streaming tokens support** in Evidor (`evidor-core`).

---

## What is Tested

1. **Token-by-Token Streaming (`stream=True`)**:
   - Synchronous streaming via `agent.send(..., stream=True)` yielding `StreamChunk` objects.
   - Asynchronous streaming via `await agent.send_async(..., stream=True)`.
   - Stringification: direct terminal printing via `print(chunk, end="", flush=True)`.
   - Accurate measurement of **Time to First Token (TTFT)** (`stream.time_to_first_token_ms`).
   - Final accumulated response (`stream.response` and `stream.get_final_response()`).

2. **Background Tool Execution During Streaming**:
   - Streaming tool arguments in the background without raw JSON polluting text output.
   - Surfacing `chunk.tool_calls` so frontend UIs can display real-time execution badges or spinners.
   - Automatic execution of tool functions and updating the conversation history with `role="tool"`.
   - Automatic continuation into follow-up turn streaming synthesized text to the caller.
   - Multi-turn conversation history integrity (`user` -> `assistant [tool_calls]` -> `tool [result]` -> `assistant [text]`).

3. **Telemetry & Observability with Langfuse**:
   - Lifecycle event emission with `is_streaming=True` on `LLMCallStartEvent` and `LLMCallEndEvent`.
   - Recording `time_to_first_token_ms` on `LLMCallEndEvent`.
   - Dual-sink verification: `InMemorySink` for programmatic assertions and `LangfuseSink` for remote observability.
   - Langfuse trace hierarchy:
     - `as_type="agent"` (`agent.run`)
     - `as_type="generation"` (`llm.<model>`) with streaming flags and TTFT metrics
     - `as_type="tool"` (`tool.<name>`) with input arguments and outputs

---

## Running the Tests

From the `evidor-testing` root:

```bash
python test_streaming_tokens.py
```

Or from within the project directory:

```bash
cd projects/streaming_tokens
python test_streaming_tokens.py
```

---

## Test Suite Components

| Test | Mode | Description |
| :--- | :--- | :--- |
| **Test 1** | Deterministic (Offline) | Scripted multi-turn mock provider and mock Langfuse client asserting chunk extraction, tool calling, TTFT, and Langfuse observation lifecycles. |
| **Test 2** | Async (Offline) | Verifies `await agent.send_async(..., stream=True)` with async iterators and `get_final_response()`. |
| **Test 3** | Live Integration | Live end-to-end execution with Google Gemini (`gemini-3.5-flash-lite`), real tool invocation (`get_stock_price`), and remote trace dispatching to Langfuse. |
