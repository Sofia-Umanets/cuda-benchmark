"""
Набор CUDA-бенчмарков для сравнения производительности native vs Docker.

Измеряет: умножение матриц, поэлементные операции, трансферы памяти,
обучение/инференс CNN, время холодного старта.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Config:
    warmup: int = 20
    iterations: int = 100
    matmul_sizes: list = field(default_factory=lambda: [512, 1024, 2048, 4096])
    batch_sizes: list = field(default_factory=lambda: [16, 32, 64, 128])
    memory_sizes_mb: list = field(default_factory=lambda: [10, 50, 100, 500])


@dataclass
class Result:
    name: str
    category: str
    mean_ms: float
    std_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    extra: dict = field(default_factory=dict)


def gpu_temp():
    """Получить температуру GPU через nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return int(out.stdout.strip())
    except Exception:
        return None


def measure(func, warmup, iterations):
    """Запустить функцию с прогревом, вернуть список времён в мс."""
    torch.cuda.synchronize()

    for _ in range(warmup):
        func()
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        func()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return times


def run_bench(name, category, func, cfg, **extra):
    """Выполнить бенчмарк и посчитать статистику."""
    times = measure(func, cfg.warmup, cfg.iterations)
    arr = np.array(times)

    return Result(
        name=name,
        category=category,
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        extra=extra
    )


def bench_matmul(cfg):
    """Бенчмарк умножения матриц (FP32/FP16)."""
    print("\n[matmul]")
    results = []
    device = torch.device("cuda")

    for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        for size in cfg.matmul_sizes:
            A = torch.randn(size, size, device=device, dtype=dtype)
            B = torch.randn(size, size, device=device, dtype=dtype)

            r = run_bench(
                f"matmul_{size}_{dtype_name}", "compute",
                lambda a=A, b=B: torch.mm(a, b), cfg,
                size=size, dtype=dtype_name
            )

            tflops = 2 * size**3 / (r.mean_ms / 1000) / 1e12
            r.extra["tflops"] = round(tflops, 2)

            print(f"  {dtype_name} {size}: {r.mean_ms:.2f}ms, {tflops:.1f} TFLOPS")
            results.append(r)

            del A, B

    torch.cuda.empty_cache()
    return results


def bench_elementwise(cfg):
    """Бенчмарк поэлементных операций."""
    print("\n[elementwise]")
    results = []
    device = torch.device("cuda")

    sizes = [1_000_000, 10_000_000, 100_000_000]
    ops = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh, "exp": torch.exp}

    for size in sizes:
        x = torch.randn(size, device=device)

        for op_name, op_fn in ops.items():
            r = run_bench(
                f"elem_{op_name}_{size}", "memory",
                lambda fn=op_fn, t=x: fn(t), cfg,
                op=op_name, size=size
            )

            bw = 2 * size * 4 / (r.mean_ms / 1000) / 1e9
            r.extra["bandwidth_gbps"] = round(bw, 1)

            print(f"  {op_name} ({size:,}): {r.mean_ms:.3f}ms, {bw:.0f} GB/s")
            results.append(r)

        del x

    torch.cuda.empty_cache()
    return results


def bench_transfer(cfg):
    """Бенчмарк трансферов host-device."""
    print("\n[transfer]")
    results = []
    device = torch.device("cuda")

    for size_mb in cfg.memory_sizes_mb:
        n = (size_mb * 1024 * 1024) // 4

        # H2D pageable
        cpu = torch.randn(n)
        r = run_bench(
            f"h2d_pageable_{size_mb}mb", "transfer",
            lambda t=cpu: t.to(device), cfg, size_mb=size_mb
        )
        bw = size_mb / 1024 / (r.mean_ms / 1000)
        r.extra["bandwidth_gbps"] = round(bw, 2)
        print(f"  h2d pageable {size_mb}MB: {r.mean_ms:.2f}ms, {bw:.1f} GB/s")
        results.append(r)

        # H2D pinned
        cpu_pin = torch.randn(n).pin_memory()
        r = run_bench(
            f"h2d_pinned_{size_mb}mb", "transfer",
            lambda t=cpu_pin: t.to(device, non_blocking=True), cfg, size_mb=size_mb
        )
        torch.cuda.synchronize()
        bw = size_mb / 1024 / (r.mean_ms / 1000)
        r.extra["bandwidth_gbps"] = round(bw, 2)
        print(f"  h2d pinned   {size_mb}MB: {r.mean_ms:.2f}ms, {bw:.1f} GB/s")
        results.append(r)

        # D2H
        gpu = torch.randn(n, device=device)
        r = run_bench(
            f"d2h_{size_mb}mb", "transfer",
            lambda t=gpu: t.cpu(), cfg, size_mb=size_mb
        )
        bw = size_mb / 1024 / (r.mean_ms / 1000)
        r.extra["bandwidth_gbps"] = round(bw, 2)
        print(f"  d2h          {size_mb}MB: {r.mean_ms:.2f}ms, {bw:.1f} GB/s")
        results.append(r)

        del cpu, cpu_pin, gpu

    torch.cuda.empty_cache()
    return results


def make_cnn():
    """Простая CNN для бенчмарков обучения/инференса."""
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, 10)
    )


def bench_training(cfg):
    """Бенчмарк шага обучения CNN."""
    print("\n[training]")
    results = []
    device = torch.device("cuda")

    model = make_cnn().to(device)
    criterion = nn.CrossEntropyLoss()
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    for bs in cfg.batch_sizes:
        x = torch.randn(bs, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (bs,), device=device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def step(m=model, o=opt, c=criterion, inp=x, tgt=y):
            o.zero_grad()
            loss = c(m(inp), tgt)
            loss.backward()
            o.step()

        try:
            r = run_bench(f"train_bs{bs}", "training", step, cfg, batch_size=bs)
            throughput = bs / (r.mean_ms / 1000)
            r.extra["samples_per_sec"] = round(throughput, 1)
            print(f"  bs={bs}: {r.mean_ms:.2f}ms, {throughput:.0f} samples/s")
            results.append(r)
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs}: OOM")
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return results


def bench_inference(cfg):
    """Бенчмарк инференса CNN."""
    print("\n[inference]")
    results = []
    device = torch.device("cuda")

    model = make_cnn().to(device).eval()

    for bs in [1, 4, 16, 64]:
        x = torch.randn(bs, 3, 32, 32, device=device)

        def forward(m=model, inp=x):
            with torch.no_grad():
                return m(inp)

        try:
            r = run_bench(f"infer_bs{bs}", "inference", forward, cfg, batch_size=bs)
            latency = r.mean_ms / bs
            r.extra["latency_per_sample_ms"] = round(latency, 3)
            print(f"  bs={bs}: {r.mean_ms:.2f}ms, {latency:.2f}ms/sample")
            results.append(r)
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs}: OOM")
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return results


def bench_cold_start(runs=20):
    """Измерение overhead инициализации CUDA в свежих процессах."""
    print("\n[cold start]")

    script = """
import torch, time, json
t = {}
t0 = time.perf_counter()
torch.cuda.init()
t["init"] = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
x = torch.zeros(1000, 1000, device="cuda")
torch.cuda.synchronize()
t["alloc"] = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
torch.mm(x, x)
torch.cuda.synchronize()
t["compute"] = (time.perf_counter() - t0) * 1000
print(json.dumps(t))
"""

    times = {"init": [], "alloc": [], "compute": []}

    for _ in range(runs):
        try:
            out = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=60
            )
            if out.returncode == 0:
                data = json.loads(out.stdout.strip())
                for k in times:
                    times[k].append(data[k])
        except Exception:
            pass

    stats = {}
    for k, v in times.items():
        if v:
            stats[k] = {"mean": round(np.mean(v), 1), "std": round(np.std(v), 1)}
            print(f"  {k}: {stats[k]['mean']:.1f} ± {stats[k]['std']:.1f}ms")

    return stats


def get_system_info():
    """Собрать информацию о системе и GPU."""
    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp": datetime.now().isoformat(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "gpu_memory_gb": round(props.total_memory / 1e9, 1),
        "gpu_temp": gpu_temp()
    }


def main():
    parser = argparse.ArgumentParser(description="Набор CUDA-бенчмарков")
    parser.add_argument("--env", required=True, choices=["native", "docker"])
    parser.add_argument("--output", help="путь для сохранения результатов")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA недоступна")

    cfg = Config()
    info = get_system_info()

    print(f"\n{'='*50}")
    print(f"CUDA Benchmark — {args.env}")
    print(f"{'='*50}")
    print(f"GPU: {info['gpu']} ({info['gpu_memory_gb']} GB)")
    print(f"PyTorch {info['pytorch']}, CUDA {info['cuda']}")

    results = []
    benchmarks = [bench_matmul, bench_elementwise, bench_transfer,
                  bench_training, bench_inference]

    for bench in benchmarks:
        try:
            results.extend(bench(cfg))
        except Exception as e:
            print(f"  ошибка: {e}")

    cold_start = bench_cold_start()

    data = {
        "env": args.env,
        "system": info,
        "config": asdict(cfg),
        "benchmarks": [asdict(r) for r in results],
        "cold_start": cold_start
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        Path("results").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"results/{args.env}_{ts}.json")

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nСохранено: {out_path}")
    print(f"Тестов: {len(results)}")


if __name__ == "__main__":
    main()