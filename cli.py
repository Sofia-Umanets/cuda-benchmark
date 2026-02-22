#!/usr/bin/env python3
"""
Единая точка входа для CUDA-бенчмарка.

Команды:
    full    - полный цикл: запуск + анализ + графики
    run     - только запуск бенчмарков
    analyze - анализ существующих результатов
    plot    - построение графиков

Примеры:
    python cli.py full --runs 5
    python cli.py analyze results/20240115_120000
    python cli.py plot results/20240115_120000
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from analysis import (
    analyze_results, 
    add_cold_start_to_df, 
    create_final_report, 
    format_report,
    MIN_RUNS_FOR_SIGNIFICANCE,
    RECOMMENDED_RUNS
)
from plotting import create_all_plots
from runner import run_all_benchmarks


def cmd_run(args):
    """Только запуск бенчмарков."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / ts

    try:
        results_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        sys.exit(f"Ошибка: нет прав на запись в {results_dir}")

    print(f"\nCUDA Benchmark")
    print(f"  Запусков: {args.runs}")
    print(f"  Результаты: {results_dir}")
    
    if args.runs < MIN_RUNS_FOR_SIGNIFICANCE:
        print(f"\n  Примечание: для статистически значимых результатов")
        print(f"  рекомендуется минимум {MIN_RUNS_FOR_SIGNIFICANCE} запусков")

    paths = run_all_benchmarks(
        results_dir=results_dir,
        runs=args.runs,
        cooldown=args.cooldown,
        cooldown_timeout=args.cooldown_timeout,
        skip_warmup=args.no_warmup
    )

    if not paths["native"] or not paths["docker"]:
        sys.exit("Ошибка: недостаточно успешных запусков")

    print(f"\nЗапуски завершены:")
    print(f"  native: {len(paths['native'])}")
    print(f"  docker: {len(paths['docker'])}")
    print(f"\nРезультаты: {results_dir}")
    print(f"Для анализа: python cli.py analyze {results_dir}")

    return results_dir


def cmd_analyze(args):
    """Анализ существующих результатов."""
    results_dir = Path(args.path)
    if not results_dir.exists():
        sys.exit(f"Ошибка: директория не найдена: {results_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    print(f"\nАнализ: {results_dir}")
    
    report, df = analyze_results(results_dir)
    df = add_cold_start_to_df(df, report)
    
    saved = create_final_report(report, df, output_dir)
    
    print(f"\nФайлы:")
    for p in saved:
        print(f"  {p}")
    
    print(format_report(report, df))

    return output_dir, df


def cmd_plot(args):
    """Построение графиков."""
    results_dir = Path(args.path)
    
    csv_path = results_dir / "comparison.csv"
    if not csv_path.exists():
        print(f"comparison.csv не найден, выполняем анализ...")
        report, df = analyze_results(results_dir)
        df = add_cold_start_to_df(df, report)
        create_final_report(report, df, results_dir)
    else:
        import pandas as pd
        df = pd.read_csv(csv_path)

    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    plots = create_all_plots(df, output_dir)
    
    print(f"\nГрафики:")
    for p in plots:
        print(f"  {p}")


def cmd_full(args):
    """Полный цикл: запуск + анализ + графики."""
    
    # Запуск
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / ts

    try:
        results_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        sys.exit(f"Ошибка: нет прав на запись в {results_dir}")

    print(f"\nCUDA Benchmark — полный цикл")
    print(f"  Запусков: {args.runs}")
    print(f"  Результаты: {results_dir}")

    paths = run_all_benchmarks(
        results_dir=results_dir,
        runs=args.runs,
        cooldown=args.cooldown,
        cooldown_timeout=args.cooldown_timeout,
        skip_warmup=args.no_warmup
    )

    if not paths["native"] or not paths["docker"]:
        sys.exit("Ошибка: недостаточно успешных запусков")

    # Анализ
    print(f"\n{'='*60}")
    print("АНАЛИЗ")
    print(f"{'='*60}")
    
    report, df = analyze_results(results_dir)
    df = add_cold_start_to_df(df, report)
    
    saved = create_final_report(report, df, results_dir)
    
    print(f"\nФайлы:")
    for p in saved:
        print(f"  {p}")
    
    print(format_report(report, df))

    # Графики
    if not args.no_plots:
        print(f"\n{'='*60}")
        print("ГРАФИКИ")
        print(f"{'='*60}")
        
        plots = create_all_plots(df, results_dir)
        print(f"\nСозданы:")
        for p in plots:
            print(f"  {p}")

    print(f"\n{'='*60}")
    print(f"Готово: {results_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="cuda-benchmark",
        description="CUDA бенчмарк: сравнение производительности native vs Docker",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="команда")

    # run
    p_run = subparsers.add_parser("run", help="запустить бенчмарки")
    p_run.add_argument("--runs", type=int, default=3, 
                       help=f"количество запусков (рекомендуется >={RECOMMENDED_RUNS})")
    p_run.add_argument("--output-dir", default="results")
    p_run.add_argument("--cooldown", type=int, default=55)
    p_run.add_argument("--cooldown-timeout", type=int, default=60)
    p_run.add_argument("--no-warmup", action="store_true")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="проанализировать результаты")
    p_analyze.add_argument("path", help="директория с результатами")
    p_analyze.add_argument("--output-dir")

    # plot
    p_plot = subparsers.add_parser("plot", help="построить графики")
    p_plot.add_argument("path", help="директория с результатами")
    p_plot.add_argument("--output-dir")

    # full
    p_full = subparsers.add_parser("full", help="полный цикл")
    p_full.add_argument("--runs", type=int, default=3,
                        help=f"количество запусков (рекомендуется >={RECOMMENDED_RUNS})")
    p_full.add_argument("--output-dir", default="results")
    p_full.add_argument("--cooldown", type=int, default=55)
    p_full.add_argument("--cooldown-timeout", type=int, default=60)
    p_full.add_argument("--no-warmup", action="store_true")
    p_full.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "run": cmd_run,
        "analyze": cmd_analyze,
        "plot": cmd_plot,
        "full": cmd_full,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()