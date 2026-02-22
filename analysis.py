"""
Модуль анализа результатов бенчмарков.

Обеспечивает:
- Агрегацию данных из нескольких запусков
- Статистическое сравнение native vs docker
- Оценку значимости различий (Welch's t-test)
- Формирование отчётов
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


MIN_RUNS_FOR_SIGNIFICANCE = 3
RECOMMENDED_RUNS = 5
SIGNIFICANCE_LEVEL = 0.05


@dataclass
class StatisticalResult:
    """Результат статистического сравнения."""
    t_statistic: Optional[float]
    p_value: Optional[float]
    significant: Optional[bool]  # None = невозможно определить
    reason: str  # почему невозможно или результат


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    """Сохранить dict в JSON с обработкой numpy-типов."""
    def convert(x):
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if hasattr(x, '__dict__'):
            return asdict(x) if hasattr(x, '__dataclass_fields__') else x.__dict__
        return x

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert, ensure_ascii=False)


def get_category(name):
    """Определить категорию бенчмарка по имени теста."""
    prefixes = {
        "matmul": "compute",
        "elem": "memory",
        "h2d": "transfer",
        "d2h": "transfer",
        "train": "training",
        "infer": "inference",
    }
    for prefix, category in prefixes.items():
        if name.startswith(prefix):
            return category
    return "other"


def welch_ttest(mean1, std1, n1, mean2, std2, n2) -> StatisticalResult:
    """
    Welch's t-test для сравнения двух выборок.
    
    Возвращает StatisticalResult с информацией о значимости
    или причиной, почему тест невозможен.
    """
    # Проверка минимального количества данных
    if n1 < MIN_RUNS_FOR_SIGNIFICANCE or n2 < MIN_RUNS_FOR_SIGNIFICANCE:
        return StatisticalResult(
            t_statistic=None,
            p_value=None,
            significant=None,
            reason=f"недостаточно данных (n1={n1}, n2={n2}, минимум={MIN_RUNS_FOR_SIGNIFICANCE})"
        )
    
    # Проверка вариации
    if std1 == 0 and std2 == 0:
        if mean1 == mean2:
            return StatisticalResult(
                t_statistic=0.0,
                p_value=1.0,
                significant=False,
                reason="значения идентичны"
            )
        else:
            return StatisticalResult(
                t_statistic=None,
                p_value=None,
                significant=None,
                reason="нулевая дисперсия при разных средних"
            )
    
    try:
        from scipy import stats
    except ImportError:
        return StatisticalResult(
            t_statistic=None,
            p_value=None,
            significant=None,
            reason="scipy не установлен"
        )
    
    # Стандартные ошибки среднего
    se1 = std1 / np.sqrt(n1) if std1 > 0 else 0
    se2 = std2 / np.sqrt(n2) if std2 > 0 else 0
    se_diff = np.sqrt(se1**2 + se2**2)
    
    if se_diff == 0:
        return StatisticalResult(
            t_statistic=None,
            p_value=None,
            significant=None,
            reason="нулевая стандартная ошибка"
        )
    
    t_stat = (mean2 - mean1) / se_diff
    
    # Степени свободы по Welch-Satterthwaite
    if se1 > 0 and se2 > 0:
        num = (se1**2 + se2**2)**2
        denom = (se1**4 / (n1 - 1)) + (se2**4 / (n2 - 1))
        df = num / denom if denom > 0 else min(n1, n2) - 1
    else:
        df = n1 + n2 - 2
    
    p_value = float(2 * stats.t.sf(abs(t_stat), df))
    significant = p_value < SIGNIFICANCE_LEVEL
    
    if significant:
        direction = "docker медленнее" if t_stat > 0 else "docker быстрее"
        reason = f"значимо ({direction})"
    else:
        reason = "различие не значимо"
    
    return StatisticalResult(
        t_statistic=round(float(t_stat), 4),
        p_value=round(p_value, 4),
        significant=significant,
        reason=reason
    )


def aggregate_runs(results):
    """Агрегировать несколько запусков в статистику."""
    by_name = {}
    for run in results:
        for bench in run["benchmarks"]:
            name = bench["name"]
            by_name.setdefault(name, []).append(bench["mean_ms"])

    benchmarks = []
    for name, times in by_name.items():
        times_arr = np.array(times)
        n = len(times_arr)
        benchmarks.append({
            "name": name,
            "category": get_category(name),
            "mean": float(np.mean(times_arr)),
            "std": float(np.std(times_arr, ddof=1)) if n > 1 else 0.0,
            "median": float(np.median(times_arr)),
            "min": float(np.min(times_arr)),
            "max": float(np.max(times_arr)),
            "values": [round(v, 4) for v in times],
            "n": n
        })

    # Cold start
    cold = {}
    for run in results:
        cs = run.get("cold_start", {})
        for key in ["init", "alloc", "compute"]:
            if key in cs:
                cold.setdefault(key, []).append(cs[key]["mean"])

    cold_agg = {}
    for k, v in cold.items():
        v_arr = np.array(v)
        n = len(v_arr)
        cold_agg[k] = {
            "mean": float(np.mean(v_arr)),
            "std": float(np.std(v_arr, ddof=1)) if n > 1 else 0.0,
            "n": n
        }

    return {"benchmarks": benchmarks, "cold_start": cold_agg, "runs": len(results)}


def compare_single_test(native_data, docker_data):
    """Сравнить один тест между native и docker."""
    n_mean, n_std, n_n = native_data["mean"], native_data["std"], native_data["n"]
    d_mean, d_std, d_n = docker_data["mean"], docker_data["std"], docker_data["n"]
    
    # Overhead
    if n_mean > 0:
        overhead = (d_mean - n_mean) / n_mean * 100
    else:
        overhead = 0.0

    # Погрешность overhead
    if n_mean > 0 and n_n > 1 and d_n > 1:
        # Стандартная ошибка разности
        se_n = n_std / np.sqrt(n_n)
        se_d = d_std / np.sqrt(d_n)
        se_diff = np.sqrt(se_n**2 + se_d**2)
        overhead_err = se_diff / n_mean * 100
    else:
        overhead_err = 0.0

    # Статистический тест
    stat_result = welch_ttest(n_mean, n_std, n_n, d_mean, d_std, d_n)

    return {
        "overhead_%": round(overhead, 2),
        "overhead_err_%": round(overhead_err, 2),
        "t_statistic": stat_result.t_statistic,
        "p_value": stat_result.p_value,
        "significant": stat_result.significant,
        "significance_reason": stat_result.reason
    }


def compare_results(native, docker):
    """Сравнить результаты native vs docker."""
    n_map = {b["name"]: b for b in native["benchmarks"]}
    d_map = {b["name"]: b for b in docker["benchmarks"]}

    rows = []
    for name in n_map:
        if name not in d_map:
            continue

        n, d = n_map[name], d_map[name]
        comparison = compare_single_test(n, d)

        rows.append({
            "name": name,
            "category": n["category"],
            "native_ms": round(n["mean"], 4),
            "native_std": round(n["std"], 4),
            "native_n": n["n"],
            "docker_ms": round(d["mean"], 4),
            "docker_std": round(d["std"], 4),
            "docker_n": d["n"],
            **comparison
        })

    return rows


def analyze_results(results_dir):
    """
    Проанализировать результаты в директории.
    
    Returns:
        tuple: (report_dict, comparison_dataframe)
    """
    results_dir = Path(results_dir)

    native_files = sorted(results_dir.glob("native_run[1-9]*.json"))
    docker_files = sorted(results_dir.glob("docker_run[1-9]*.json"))

    if not native_files or not docker_files:
        raise FileNotFoundError(f"Нет результатов в {results_dir}")

    native_data = [load_json(p) for p in native_files]
    docker_data = [load_json(p) for p in docker_files]

    native_agg = aggregate_runs(native_data)
    docker_agg = aggregate_runs(docker_data)
    comparison = compare_results(native_agg, docker_agg)

    report = {
        "timestamp": results_dir.name,
        "runs": {
            "native": len(native_files),
            "docker": len(docker_files),
            "total": len(native_files) + len(docker_files)
        },
        "config": {
            "min_runs_for_significance": MIN_RUNS_FOR_SIGNIFICANCE,
            "recommended_runs": RECOMMENDED_RUNS,
            "significance_level": SIGNIFICANCE_LEVEL
        },
        "system": native_data[0].get("system", {}),
        "native": native_agg,
        "docker": docker_agg,
        "comparison": comparison
    }

    return report, pd.DataFrame(comparison)


def add_cold_start_to_df(df, report):
    """Добавить метрики cold start в DataFrame."""
    native_cs = report["native"].get("cold_start", {})
    docker_cs = report["docker"].get("cold_start", {})

    rows = []
    for key in ["init", "alloc", "compute"]:
        if key not in native_cs or key not in docker_cs:
            continue
            
        n = native_cs[key]
        d = docker_cs[key]
        
        comparison = compare_single_test(n, d)

        rows.append({
            "name": f"cold_start_{key}",
            "category": "cold_start",
            "native_ms": round(n["mean"], 4),
            "native_std": round(n["std"], 4),
            "native_n": n["n"],
            "docker_ms": round(d["mean"], 4),
            "docker_std": round(d["std"], 4),
            "docker_n": d["n"],
            **comparison
        })

    if rows:
        return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df


def compute_summary(df, runs_info):
    """Вычислить сводную статистику."""
    overhead = df["overhead_%"]
    
    # Группировка по значимости
    sig_true = df[df["significant"] == True]
    sig_false = df[df["significant"] == False]
    sig_unknown = df[df["significant"].isna()]
    
    docker_slower = sig_true[sig_true["overhead_%"] > 0]
    docker_faster = sig_true[sig_true["overhead_%"] < 0]
    
    can_compute = (runs_info["native"] >= MIN_RUNS_FOR_SIGNIFICANCE and
                   runs_info["docker"] >= MIN_RUNS_FOR_SIGNIFICANCE)
    
    return {
        "tests_total": len(df),
        "overhead": {
            "mean": round(float(overhead.mean()), 2),
            "median": round(float(overhead.median()), 2),
            "std": round(float(overhead.std()), 2),
            "min": round(float(overhead.min()), 2),
            "max": round(float(overhead.max()), 2)
        },
        "significance": {
            "can_compute": can_compute,
            "docker_slower": len(docker_slower),
            "docker_faster": len(docker_faster),
            "no_difference": len(sig_false),
            "unknown": len(sig_unknown)
        },
        "by_category": compute_summary_by_category(df).to_dict(orient="index")
    }


def compute_summary_by_category(df):
    """Статистика по категориям."""
    return df.groupby("category")["overhead_%"].agg([
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("max", "max"),
        ("count", "count")
    ]).round(2)


def create_final_report(report, df, output_dir):
    """
    Создать итоговый отчёт со всеми результатами.
    
    Сохраняет:
    - report.json: полный отчёт
    - comparison.csv: детальное сравнение
    - summary.json: краткая сводка для быстрого просмотра
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    runs_info = report["runs"]
    summary = compute_summary(df, runs_info)
    
    # Определяем общий вывод
    conclusion = make_conclusion(summary, runs_info)
    
    # Итоговый summary.json
    summary_data = {
        "timestamp": report["timestamp"],
        "system": {
            "gpu": report["system"].get("gpu", "N/A"),
            "cuda": report["system"].get("cuda", "N/A"),
            "pytorch": report["system"].get("pytorch", "N/A")
        },
        "runs": runs_info,
        "summary": summary,
        "conclusion": conclusion,
        "significant_tests": get_significant_tests(df)
    }
    
    paths = []
    
    # summary.json
    summary_path = output_dir / "summary.json"
    save_json(summary_data, summary_path)
    paths.append(summary_path)
    
    # report.json (полный)
    report["summary"] = summary
    report["conclusion"] = conclusion
    report_path = output_dir / "report.json"
    save_json(report, report_path)
    paths.append(report_path)
    
    # comparison.csv
    csv_path = output_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    paths.append(csv_path)
    
    # by_category.csv
    cat_path = output_dir / "by_category.csv"
    compute_summary_by_category(df).to_csv(cat_path)
    paths.append(cat_path)
    
    return paths


def make_conclusion(summary, runs_info):
    """Сформировать вывод на основе статистики."""
    sig = summary["significance"]
    overhead = summary["overhead"]
    
    can_compute = sig["can_compute"]
    n_runs = runs_info["native"]
    d_runs = runs_info["docker"]
    
    if not can_compute:
        return {
            "status": "insufficient_data",
            "message": f"Недостаточно данных для статистических выводов",
            "details": f"Выполнено запусков: native={n_runs}, docker={d_runs}. "
                       f"Минимум для анализа: {MIN_RUNS_FOR_SIGNIFICANCE}. "
                       f"Рекомендуется: {RECOMMENDED_RUNS}.",
            "recommendation": f"Запустите: python cli.py full --runs {RECOMMENDED_RUNS}"
        }
    
    docker_slower = sig["docker_slower"]
    docker_faster = sig["docker_faster"]
    no_diff = sig["no_difference"]
    total = summary["tests_total"]
    
    # Определяем статус
    if docker_slower == 0 and docker_faster == 0:
        status = "no_difference"
        message = "Статистически значимых различий не обнаружено"
        details = f"Все {no_diff} тестов показали отсутствие значимой разницы (p >= {SIGNIFICANCE_LEVEL})"
    elif docker_slower > docker_faster:
        status = "docker_slower"
        message = f"Docker медленнее в {docker_slower} из {total} тестов"
        details = f"Средний overhead: {overhead['mean']:+.2f}%"
    elif docker_faster > docker_slower:
        status = "docker_faster"
        message = f"Docker быстрее в {docker_faster} из {total} тестов"
        details = f"Средний overhead: {overhead['mean']:+.2f}%"
    else:
        status = "mixed"
        message = "Результаты неоднозначны"
        details = f"Docker медленнее в {docker_slower} тестах, быстрее в {docker_faster}"
    
    return {
        "status": status,
        "message": message,
        "details": details,
        "overhead_mean": overhead["mean"],
        "overhead_median": overhead["median"]
    }


def get_significant_tests(df, limit=10):
    """Получить список значимых тестов, отсортированных по величине эффекта."""
    sig = df[df["significant"] == True].copy()
    if len(sig) == 0:
        return []
    
    sig = sig.reindex(sig["overhead_%"].abs().sort_values(ascending=False).index)
    
    result = []
    for _, row in sig.head(limit).iterrows():
        result.append({
            "name": row["name"],
            "category": row["category"],
            "overhead_%": row["overhead_%"],
            "p_value": row["p_value"],
            "native_ms": row["native_ms"],
            "docker_ms": row["docker_ms"]
        })
    
    return result


def format_report(report, df):
    """Сформировать текстовый отчёт для вывода в консоль."""
    lines = []
    
    sys_info = report.get("system", {})
    runs_info = report.get("runs", {})
    summary = compute_summary(df, runs_info)
    conclusion = make_conclusion(summary, runs_info)
    
    n_runs = runs_info.get("native", 0)
    d_runs = runs_info.get("docker", 0)
    overhead = summary["overhead"]
    sig = summary["significance"]
    
    # Заголовок
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"РЕЗУЛЬТАТЫ БЕНЧМАРКА")
    lines.append("=" * 60)
    
    # Информация о запусках
    lines.append(f"\nЗапусков: native={n_runs}, docker={d_runs}")
    
    # Предупреждение о недостаточных данных
    if not sig["can_compute"]:
        lines.append("")
        lines.append("-" * 60)
        lines.append("ВНИМАНИЕ: недостаточно данных для статистических выводов")
        lines.append(f"  Минимум: {MIN_RUNS_FOR_SIGNIFICANCE} запусков каждого типа")
        lines.append(f"  Рекомендуется: {RECOMMENDED_RUNS} запусков")
        lines.append(f"  Команда: python cli.py full --runs {RECOMMENDED_RUNS}")
        lines.append("-" * 60)
    
    # Система
    if sys_info:
        lines.append(f"\nСистема:")
        lines.append(f"  GPU: {sys_info.get('gpu', 'N/A')}")
        lines.append(f"  CUDA: {sys_info.get('cuda', 'N/A')}")
        lines.append(f"  PyTorch: {sys_info.get('pytorch', 'N/A')}")
    
    # Overhead
    qualifier = " (предварительно)" if not sig["can_compute"] else ""
    lines.append(f"\nOverhead Docker{qualifier}:")
    lines.append(f"  Среднее:  {overhead['mean']:+.2f}%")
    lines.append(f"  Медиана:  {overhead['median']:+.2f}%")
    lines.append(f"  Std:      {overhead['std']:.2f}%")
    lines.append(f"  Диапазон: [{overhead['min']:+.1f}%, {overhead['max']:+.1f}%]")
    
    # По категориям
    lines.append(f"\nПо категориям:")
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        cat_oh = cat_df["overhead_%"]
        mean_val = cat_oh.mean()
        std_val = cat_oh.std()
        lines.append(f"  {cat:12s}: {mean_val:+6.2f}% (std={std_val:.2f}%)")
    
    # Статистика
    lines.append(f"\nТестов: {summary['tests_total']}")
    
    if sig["can_compute"]:
        lines.append(f"\nСтатистическая значимость (p < {SIGNIFICANCE_LEVEL}):")
        lines.append(f"  Docker медленнее:  {sig['docker_slower']}")
        lines.append(f"  Docker быстрее:    {sig['docker_faster']}")
        lines.append(f"  Нет разницы:       {sig['no_difference']}")
        
        # Топ значимых
        significant_tests = get_significant_tests(df, limit=5)
        if significant_tests:
            lines.append(f"\nЗначимые различия:")
            for t in significant_tests:
                lines.append(f"  {t['name']:30s}: {t['overhead_%']:+6.2f}% (p={t['p_value']:.4f})")
    
    # Вывод
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"ВЫВОД: {conclusion['message']}")
    if conclusion.get("details"):
        lines.append(f"  {conclusion['details']}")
    if conclusion.get("recommendation"):
        lines.append(f"  {conclusion['recommendation']}")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# Обратная совместимость
def save_analysis(report, df, output_dir):
    """Обёртка для обратной совместимости."""
    return create_final_report(report, df, output_dir)