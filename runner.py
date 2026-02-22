"""
Модуль запуска бенчмарков.
"""

import os
import random
import subprocess
import sys
import time
from pathlib import Path


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


def wait_cooldown(target, max_wait):
    """Подождать охлаждения GPU до целевой температуры."""
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
    """Сбросить файловый кэш (требует passwordless sudo)."""
    try:
        subprocess.run(["sync"], capture_output=True, timeout=10)
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True, timeout=10
        )
    except Exception:
        pass


def run_native(output):
    """Запустить бенчмарк нативно."""
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    result = subprocess.run(
        [sys.executable, "benchmark.py", "--env", "native", "--output", str(output)],
        env=env
    )
    return result.returncode == 0


def run_docker(output, results_dir):
    """Запустить бенчмарк в Docker-контейнере."""
    result = subprocess.run([
        "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
        "-v", f"{results_dir.absolute()}:/results",
        "cuda-benchmark",
        "--env", "docker", "--output", f"/results/{output.name}"
    ])
    return result.returncode == 0


def run_single(env, run_id, results_dir, cooldown_target, cooldown_max):
    """Выполнить один запуск бенчмарка."""
    print(f"\n  [{env}] run {run_id}")
    drop_caches()
    wait_cooldown(cooldown_target, cooldown_max)

    output = results_dir / f"{env}_run{run_id}.json"
    
    if env == "native":
        success = run_native(output)
    else:
        success = run_docker(output, results_dir)

    if success and output.exists():
        return output

    print("    ошибка")
    return None


def run_all_benchmarks(results_dir, runs, cooldown, cooldown_timeout, skip_warmup=False):
    """
    Запустить все бенчмарки в случайном порядке.
    
    Args:
        results_dir: директория для результатов
        runs: количество запусков каждого типа
        cooldown: целевая температура GPU
        cooldown_timeout: максимальное время ожидания охлаждения
        skip_warmup: пропустить прогревочный запуск
    
    Returns:
        dict с ключами "native" и "docker", значения — списки путей к результатам
    """
    results_dir = Path(results_dir)
    
    # Формируем план запусков
    plan = []
    for i in range(1, runs + 1):
        plan.append(("native", i))
        plan.append(("docker", i))
    random.shuffle(plan)

    print(f"\nПорядок: {' -> '.join(f'{e[0]}#{e[1]}' for e in plan)}")

    # Прогрев
    if not skip_warmup:
        print("\nПрогрев...")
        warmup = run_single("native", 0, results_dir, cooldown, cooldown_timeout)
        if warmup:
            warmup.unlink(missing_ok=True)

    # Основные запуски
    print(f"\nЗапуски ({len(plan)} всего):")
    paths = {"native": [], "docker": []}

    for env, run_id in plan:
        path = run_single(env, run_id, results_dir, cooldown, cooldown_timeout)
        if path:
            paths[env].append(path)

    return paths