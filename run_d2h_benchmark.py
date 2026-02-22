#!/usr/bin/env python3
"""
Узконаправленный бенчмарк D2H (device-to-host) трансферов.
Для детального исследования производительности трансферов.

Использование:
    python run_d2h_benchmark.py --runs 10 --size 1024 --iterations 500
"""

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def gpu_temp():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return int(out.stdout.strip())
    except Exception:
        return None


def wait_cooldown(target, max_wait):
    temp = gpu_temp()
    if temp is None:
        time.sleep(10)
        return
    if temp <= target:
        print(f"    {temp}°C", flush=True)
        time.sleep(3)
        return
    waited = 0
    while temp > target and waited < max_wait:
        print(f"    {temp}°C, ждём...", flush=True)
        time.sleep(10)
        waited += 10
        temp = gpu_temp()
        if temp is None:
            break
    time.sleep(3)


def drop_caches():
    try:
        subprocess.run(["sync"], capture_output=True, timeout=10)
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True, timeout=10
        )
    except Exception:
        pass


def measure_d2h(size_mb, iterations, warmup=20):
    """Измерить время D2H трансфера для заданного размера."""
    device = torch.device("cuda")
    n = (size_mb * 1024 * 1024) // 4

    gpu_tensor = torch.randn(n, device=device)

    def copy_to_cpu():
        return gpu_tensor.cpu()

    torch.cuda.synchronize()
    for _ in range(warmup):
        copy_to_cpu()
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        copy_to_cpu()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    del gpu_tensor
    torch.cuda.empty_cache()

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "values_ms": times.tolist()
    }


def run_native(output_path, size_mb, iterations):
    print(f"\n  [native] запуск")
    drop_caches()
    result = measure_d2h(size_mb, iterations)
    data = {
        "env": "native",
        "timestamp": datetime.now().isoformat(),
        "size_mb": size_mb,
        "iterations": iterations,
        **result
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return True


def run_docker(output_path, size_mb, iterations, results_dir):
    print(f"\n  [docker] запуск")
    drop_caches()
    cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
        "-v", f"{Path(__file__).parent.absolute()}:/workspace",
        "-v", f"{results_dir.absolute()}:/results",
        "--entrypoint", "python3",
        "cuda-benchmark",
        "/workspace/run_d2h_benchmark.py",
        "--inside-docker",
        "--size", str(size_mb),
        "--iterations", str(iterations),
        "--output", f"/results/{output_path.name}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ошибка: {result.stderr}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Бенчмарк D2H трансферов")
    parser.add_argument("--runs", type=int, default=10, help="запусков на окружение")
    parser.add_argument("--size", type=int, default=500, help="размер данных в МБ")
    parser.add_argument("--iterations", type=int, default=500, help="итераций на запуск")
    parser.add_argument("--cooldown", type=int, default=55, help="целевая температура")
    parser.add_argument("--cooldown-timeout", type=int, default=60)
    parser.add_argument("--output-dir", default="d2h_results")
    parser.add_argument("--inside-docker", action="store_true", help="внутренний флаг")
    parser.add_argument("--output", help="путь вывода (для inside-docker)")
    args = parser.parse_args()

    # Запуск внутри контейнера
    if args.inside_docker:
        if not args.output:
            sys.exit("--output обязателен с --inside-docker")
        result = measure_d2h(args.size, args.iterations)
        data = {
            "env": "docker",
            "timestamp": datetime.now().isoformat(),
            "size_mb": args.size,
            "iterations": args.iterations,
            **result
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        return

    # Основной режим: серия запусков native и docker
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / ts
    results_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for i in range(1, args.runs + 1):
        plan.append(("native", i))
        plan.append(("docker", i))
    random.shuffle(plan)

    print(f"\nПорядок: {' -> '.join(f'{e[0]}#{e[1]}' for e in plan)}")

    # Прогрев
    print("\nПрогрев...")
    warmup_out = results_dir / "warmup_native.json"
    run_native(warmup_out, args.size, args.iterations)
    warmup_out.unlink()

    paths = {"native": [], "docker": []}

    for env, run_id in plan:
        print(f"\n--- {env} run {run_id} ---")
        drop_caches()
        wait_cooldown(args.cooldown, args.cooldown_timeout)

        out_file = results_dir / f"{env}_run{run_id}.json"
        if env == "native":
            success = run_native(out_file, args.size, args.iterations)
        else:
            success = run_docker(out_file, args.size, args.iterations, results_dir)

        if success:
            paths[env].append(out_file)
            print(f"    сохранено: {out_file}")
        else:
            print("    ошибка, пропускаем")

    # Сбор результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)

    native_means, docker_means = [], []
    native_all, docker_all = [], []

    for p in paths["native"]:
        with open(p) as f:
            data = json.load(f)
            native_means.append(data["mean_ms"])
            native_all.extend(data["values_ms"])

    for p in paths["docker"]:
        with open(p) as f:
            data = json.load(f)
            docker_means.append(data["mean_ms"])
            docker_all.extend(data["values_ms"])

    if not native_means or not docker_means:
        sys.exit("Недостаточно данных")

    native_all = np.array(native_all)
    docker_all = np.array(docker_all)
    overhead = (docker_all.mean() - native_all.mean()) / native_all.mean() * 100

    print(f"\nРазмер данных: {args.size} МБ")
    print(f"Запусков на окружение: {len(native_means)}")
    print(f"Итераций на запуск: {args.iterations}")

    print(f"\nСредние по запускам:")
    print(f"  native: {np.mean(native_means):.3f} ms (std: {np.std(native_means):.3f})")
    print(f"  docker: {np.mean(docker_means):.3f} ms (std: {np.std(docker_means):.3f})")

    print(f"\nВсе измерения вместе:")
    print(f"  native: {native_all.mean():.3f} ± {native_all.std():.3f} ms")
    print(f"  docker: {docker_all.mean():.3f} ± {docker_all.std():.3f} ms")
    print(f"  overhead: {overhead:+.2f}%")

    try:
        from scipy import stats
        _, p_value = stats.ttest_ind(native_all, docker_all, equal_var=False)
        print(f"\nt-test Уэлча p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("Разница статистически значима (p < 0.05)")
        else:
            print("Разница статистически незначима (p >= 0.05)")
    except ImportError:
        p_value = None
        print("\nscipy не установлен, t-test пропущен")

    summary = {
        "size_mb": args.size,
        "runs": len(native_means),
        "iterations_per_run": args.iterations,
        "native": {
            "mean_of_means": float(np.mean(native_means)),
            "std_of_means": float(np.std(native_means)),
            "all_mean": float(native_all.mean()),
            "all_std": float(native_all.std()),
        },
        "docker": {
            "mean_of_means": float(np.mean(docker_means)),
            "std_of_means": float(np.std(docker_means)),
            "all_mean": float(docker_all.mean()),
            "all_std": float(docker_all.std()),
        },
        "overhead_percent": overhead,
        "p_value": p_value
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nСводка сохранена: {summary_path}")
    print(f"Директория результатов: {results_dir}")


if __name__ == "__main__":
    main()



