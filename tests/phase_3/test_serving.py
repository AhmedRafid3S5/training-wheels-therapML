"""Phase 3 · Task 4 — Model Serving & Load Testing

Tests validate:

* ``run_stream_decode``: your naive PyTorch serving loop streams tokens lazily
  (one decoding step per ``next()``) and yields the same tokens as
  ``run_decode``,
* ``run_load_test``: against a fake streaming server with known timings, it
  measures Time to First Token, total latency, tokens/sec and aggregate
  throughput correctly, and really keeps ``concurrency`` requests in flight,
* a serving benchmark table for 10 concurrent requests against the naive
  PyTorch loop and, *optionally*, against your vLLM / TGI server.

To include your server, start it and run e.g.::

    PHASE3_SERVER_URL=http://localhost:8000 PHASE3_SERVER_MODEL=microsoft/Phi-3-mini-4k-instruct \\
        pytest tests/phase_3/test_serving.py -v -k Server

The server test is skipped when ``PHASE3_SERVER_URL`` is not set.
"""

import inspect
import os
import threading
import time

import pytest
import torch

from tests.adapters_0 import get_server_client, run_decode, run_load_test, run_stream_decode
from tests.phase_3.common import SMALL_CONFIG, BigramModel

# ============================================================================
# Streaming generation (naive PyTorch loop)
# ============================================================================


class TestStreamDecode:
    def test_is_lazy(self):
        model = BigramModel(20)
        stream = run_stream_decode(model, [1, 2], 5, temperature=1.0, context_length=16, use_kv_cache=False)
        assert inspect.isgenerator(stream) or hasattr(stream, "__next__")
        assert model.num_calls == 0, "The model was called before the first token was requested"
        next(stream)
        assert model.num_calls == 1
        next(stream)
        assert model.num_calls == 2
        assert len(list(stream)) == 3

    @pytest.mark.parametrize("kw", [{"temperature": 0.0}, {"temperature": 0.9, "top_k": 10, "top_p": 0.9}])
    def test_same_tokens_as_decode(self, kw):
        model = BigramModel(50, seed=1)
        args = dict(kw, context_length=16, use_kv_cache=False)
        torch.manual_seed(0)
        expected = run_decode(model, [3, 4], 30, **args)
        torch.manual_seed(0)
        assert [int(t) for t in run_stream_decode(model, [3, 4], 30, **args)] == expected

    def test_kv_cache_streaming_with_mini_llama(self, small_llama):
        args = {"temperature": 0.0, "context_length": SMALL_CONFIG["context_length"], "use_kv_cache": True}
        streamed = [int(t) for t in run_stream_decode(small_llama, [1, 2, 3], 25, **args)]
        assert streamed == run_decode(small_llama, [1, 2, 3], 25, **args)


# ============================================================================
# Load-testing script against a fake server with known timings
# ============================================================================


class FakeStreamingServer:
    """``send_request(prompt)`` streams ``n`` tokens: the first after ``ttft``
    seconds, then one every ``dt`` seconds. Tracks how many requests are in
    flight at once."""

    def __init__(self, specs: dict[str, tuple[float, int, float]]):
        self.specs = specs
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def expected_latency(self, prompt):
        ttft, n, dt = self.specs[prompt]
        return ttft + (n - 1) * dt

    def __call__(self, prompt):
        ttft, n, dt = self.specs[prompt]
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(ttft)
            yield "tok"
            for _ in range(n - 1):
                time.sleep(dt)
                yield "tok"
        finally:
            with self.lock:
                self.in_flight -= 1


# Sleep overshoot and thread scheduling only ever make things *slower*.
SLACK = 0.06


def _uniform(n_prompts, ttft=0.1, n=10, dt=0.02):
    return {f"prompt {i}": (ttft, n, dt) for i in range(n_prompts)}


class TestLoadTest:
    def test_per_request_metrics(self):
        specs = {
            "short": (0.05, 5, 0.01),
            "slow start": (0.30, 5, 0.01),
            "long": (0.05, 30, 0.01),
            "one token": (0.10, 1, 0.0),
        }
        server = FakeStreamingServer(specs)
        prompts = list(specs)
        result = run_load_test(server, prompts, concurrency=4)
        assert len(result["requests"]) == len(prompts)
        print("\n  prompt        ttft    latency  tokens  tok/s")
        for prompt, r in zip(prompts, result["requests"]):
            ttft, n, _ = specs[prompt]
            latency = server.expected_latency(prompt)
            print(
                f"  {prompt:<12} {r['ttft']:.3f}   {r['latency']:.3f}  {r['num_tokens']:>6}  {r['tokens_per_sec']:.1f}"
            )
            assert r["num_tokens"] == n
            assert ttft - 0.005 <= r["ttft"] <= ttft + SLACK, f"{prompt}: TTFT {r['ttft']:.3f}s, expected ≈{ttft}s"
            assert latency - 0.005 <= r["latency"] <= latency + SLACK, (
                f"{prompt}: latency {r['latency']:.3f}s, expected ≈{latency:.3f}s"
            )
            assert r["tokens_per_sec"] == pytest.approx(r["num_tokens"] / r["latency"], rel=1e-6)

    def test_ttft_is_measured_at_first_token(self):
        """A long stream with a fast first token: TTFT must not equal the latency."""
        server = FakeStreamingServer({"p": (0.02, 20, 0.02)})
        r = run_load_test(server, ["p"], concurrency=1)["requests"][0]
        assert r["ttft"] < 0.1 < r["latency"]

    @pytest.mark.parametrize("concurrency, waves", [(10, 1), (5, 2), (1, 10)])
    def test_concurrency(self, concurrency, waves):
        server = FakeStreamingServer(_uniform(10))
        prompts = list(server.specs)
        one = server.expected_latency(prompts[0])  # 0.28 s
        result = run_load_test(server, prompts, concurrency=concurrency)
        print(f"\nconcurrency={concurrency}: total_time={result['total_time']:.3f}s (ideal {waves * one:.2f}s)")
        assert server.max_in_flight == concurrency, (
            f"Expected {concurrency} requests in flight at once, saw at most {server.max_in_flight}"
        )
        assert waves * one - 0.01 <= result["total_time"] <= waves * (one + SLACK)

    def test_queueing_time_not_counted_in_ttft(self):
        server = FakeStreamingServer(_uniform(6, ttft=0.1, n=3, dt=0.02))
        result = run_load_test(server, list(server.specs), concurrency=2)
        ttfts = [r["ttft"] for r in result["requests"]]
        assert max(ttfts) <= 0.1 + SLACK, f"TTFT should be measured from when the request is sent: {ttfts}"
        assert result["total_time"] >= 3 * 0.14 - 0.01

    def test_aggregate_throughput(self):
        server = FakeStreamingServer(_uniform(8, n=12))
        result = run_load_test(server, list(server.specs), concurrency=4)
        total_tokens = sum(r["num_tokens"] for r in result["requests"])
        assert total_tokens == 8 * 12
        assert result["throughput"] == pytest.approx(total_tokens / result["total_time"], rel=1e-6)
        # The wall-clock window covers every request but is shorter than running them back to back.
        assert result["total_time"] >= max(r["latency"] for r in result["requests"]) - 1e-6
        assert result["total_time"] < sum(r["latency"] for r in result["requests"])


# ============================================================================
# Serving benchmark: naive PyTorch loop (and optionally vLLM / TGI)
# ============================================================================

N_REQUESTS, MAX_NEW_TOKENS = 10, 32
PROMPTS = [f"Request {i}: To be, or not to be" for i in range(N_REQUESTS)]


def _print_table(title, result):
    reqs = result["requests"]
    mean = lambda key: sum(r[key] for r in reqs) / len(reqs)
    print(f"\n{title}")
    print("  metric                     mean      max")
    print(f"  TTFT (s)               {mean('ttft'):>8.3f} {max(r['ttft'] for r in reqs):>8.3f}")
    print(f"  latency (s)            {mean('latency'):>8.3f} {max(r['latency'] for r in reqs):>8.3f}")
    print(f"  tokens/sec per request {mean('tokens_per_sec'):>8.1f}")
    print(f"  aggregate throughput   {result['throughput']:>8.1f} tok/s   ({result['total_time']:.2f}s total)")


def _check_sane(result, expected_tokens=None):
    assert len(result["requests"]) == N_REQUESTS
    for r in result["requests"]:
        assert r["num_tokens"] > 0
        if expected_tokens is not None:
            assert r["num_tokens"] == expected_tokens
        assert 0 < r["ttft"] <= r["latency"]
    assert result["throughput"] > 0


class TestServingBenchmark:
    def test_naive_pytorch_serving(self, small_llama):
        """One model, one KV-cache: a naive server handles one request at a
        time, so concurrent requests queue up (TTFT grows). This is the
        baseline that vLLM / TGI continuous batching improves on."""
        vocab = SMALL_CONFIG["vocab_size"]
        lock = threading.Lock()

        def send_request(prompt):
            ids = [b % vocab for b in prompt.encode()]
            with lock, torch.no_grad():
                yield from run_stream_decode(
                    small_llama,
                    ids,
                    MAX_NEW_TOKENS,
                    temperature=0.0,
                    context_length=SMALL_CONFIG["context_length"],
                    use_kv_cache=True,
                )

        result = run_load_test(send_request, PROMPTS, concurrency=N_REQUESTS)
        _print_table(f"Naive PyTorch loop — {N_REQUESTS} concurrent requests, {MAX_NEW_TOKENS} new tokens", result)
        _check_sane(result, expected_tokens=MAX_NEW_TOKENS)


@pytest.mark.skipif(not os.environ.get("PHASE3_SERVER_URL"), reason="PHASE3_SERVER_URL not set (no vLLM/TGI server)")
class TestServerBenchmark:
    def test_server_load(self):
        url = os.environ["PHASE3_SERVER_URL"]
        model = os.environ.get("PHASE3_SERVER_MODEL", "")
        send_request = get_server_client(url, model, max_tokens=64)
        run_load_test(send_request, PROMPTS[:1], concurrency=1)  # warm-up
        result = run_load_test(send_request, PROMPTS, concurrency=N_REQUESTS)
        _print_table(f"Server at {url} ({model}) — {N_REQUESTS} concurrent requests, ≤64 new tokens", result)
        _check_sane(result)
