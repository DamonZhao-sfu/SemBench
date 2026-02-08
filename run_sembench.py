#!/usr/bin/env python3
"""
SemBench Runner Script
Executes all semBench queries sequentially, evaluates quality, and generates a summary report.
"""

import os
import sys
import subprocess
import time
import re
import glob
import csv
from datetime import datetime
from pathlib import Path

# Configuration
BASE_DIR = Path("/scratch/hpc-prf-haqc/haikai/AI_UDF_private/omnimcp/omnimcp/server")

# Define query configurations: (query_file, output_path_pattern, quality_file)
# output_path_pattern is used to find the output directory
QUERY_CONFIGS = [
    {
        "name": "q1",
        "query_file": "semBench_q1.py",
        "output_dir": "q1_result.csv",
        "quality_file": None,  # No quality file for q1
    },
    {
        "name": "q2a",
        "query_file": "semBench_q2a.py",
        "output_dir": "q2a.csv",
        "quality_file": None,  # No quality file
    },
    {
        "name": "q3a",
        "query_file": "semBench_q3a.py",
        "output_dir": "q3a.csv",
        "quality_file": "semBench_q3a_quality.py",
    },
    {
        "name": "q3f",
        "query_file": "semBench_q3f.py",
        "output_dir": "q3f.csv",
        "quality_file": "semBench_q3a_quality.py",  # Uses same quality as q3a
    },
    {
        "name": "q4",
        "query_file": "semBench_q4.py",
        "output_dir": "q4.csv",
        "quality_file": "semBench_q4_quality.py",
    },
    {
        "name": "q5",
        "query_file": "semBench_q5.py",
        "output_dir": "q5.csv",
        "quality_file": "semBench_q5_quality.py",
    },
    {
        "name": "q6a",
        "query_file": "semBench_q6a.py",
        "output_dir": "q6a.csv",
        "quality_file": "semBench_q6a_quality.py",
    },
    {
        "name": "q6b",
        "query_file": "semBench_q6b.py",
        "output_dir": "q6b.csv",
        "quality_file": "semBench_q6b_quality.py",
    },
    {
        "name": "q6c",
        "query_file": "semBench_q6c.py",
        "output_dir": "q6c.csv",
        "quality_file": "semBench_q6c_quality.py",
    },
    {
        "name": "q7",
        "query_file": "semBench_q7.py",
        "output_dir": "q7.csv",
        "quality_file": "semBench_q7_quality.py",
    },
]


def find_part_file(output_dir: Path) -> str:
    """Find the part-*.csv file in a Spark output directory."""
    if not output_dir.exists():
        return None

    # Look for part-*.csv files
    part_files = list(output_dir.glob("part-*.csv"))
    if part_files:
        # Return the first one (usually there's only one with coalesce(1))
        return str(part_files[0])

    # If it's a single CSV file (not a directory)
    if output_dir.is_file() and str(output_dir).endswith('.csv'):
        return str(output_dir)

    return None


def parse_latency_from_output(output: str) -> float:
    """Parse latency from query execution output."""
    # Look for patterns like "completed in X.XX seconds" or "X.XX seconds"
    patterns = [
        r"completed in (\d+\.?\d*) seconds",
        r"Process completed in (\d+\.?\d*) seconds",
        r"Semantic join completed in (\d+\.?\d*) seconds",
        r"(\d+\.?\d*) seconds",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return -1.0  # Return -1 if not found


def parse_quality_metrics(output: str) -> dict:
    """Parse precision, recall, F1 from quality evaluation output."""
    metrics = {
        "precision": -1.0,
        "recall": -1.0,
        "f1_score": -1.0,
    }

    # Parse precision
    precision_match = re.search(r"Precision:\s*(\d+\.?\d*)", output)
    if precision_match:
        metrics["precision"] = float(precision_match.group(1))

    # Parse recall
    recall_match = re.search(r"Recall:\s*(\d+\.?\d*)", output)
    if recall_match:
        metrics["recall"] = float(recall_match.group(1))

    # Parse F1
    f1_match = re.search(r"F1 Score:\s*(\d+\.?\d*)", output)
    if f1_match:
        metrics["f1_score"] = float(f1_match.group(1))

    return metrics


def run_query(config: dict) -> dict:
    """Run a single query and return results."""
    query_name = config["name"]
    query_file = BASE_DIR / config["query_file"]
    output_dir = BASE_DIR / config["output_dir"]
    quality_file = BASE_DIR / config["quality_file"] if config["quality_file"] else None

    result = {
        "query": query_name,
        "status": "pending",
        "latency_seconds": -1.0,
        "precision": -1.0,
        "recall": -1.0,
        "f1_score": -1.0,
        "error": None,
    }

    print(f"\n{'='*60}")
    print(f"Running Query: {query_name}")
    print(f"Query File: {query_file}")
    print(f"{'='*60}")

    # Check if query file exists
    if not query_file.exists():
        result["status"] = "error"
        result["error"] = f"Query file not found: {query_file}"
        print(f"ERROR: {result['error']}")
        return result

    # Run the query
    try:
        start_time = time.time()
        process = subprocess.run(
            [sys.executable, str(query_file)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        elapsed_time = time.time() - start_time

        # Combine stdout and stderr
        full_output = process.stdout + "\n" + process.stderr
        print(full_output)

        if process.returncode != 0:
            result["status"] = "failed"
            result["error"] = f"Query returned non-zero exit code: {process.returncode}"
            result["latency_seconds"] = elapsed_time
            return result

        # Parse latency from output, or use elapsed time
        parsed_latency = parse_latency_from_output(full_output)
        result["latency_seconds"] = parsed_latency if parsed_latency > 0 else elapsed_time
        result["status"] = "completed"

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Query execution timed out (1 hour)"
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result

    # Run quality evaluation if quality file exists
    if quality_file and quality_file.exists():
        print(f"\nRunning Quality Evaluation: {quality_file.name}")

        # Find the part file
        part_file = find_part_file(output_dir)
        if not part_file:
            result["error"] = f"Output part file not found in {output_dir}"
            print(f"WARNING: {result['error']}")
        else:
            try:
                # We need to modify the quality file call to use the correct part file
                # Read the quality file and modify it to use our part file
                quality_output = run_quality_evaluation(quality_file, part_file)
                print(quality_output)

                # Parse metrics
                metrics = parse_quality_metrics(quality_output)
                result["precision"] = metrics["precision"]
                result["recall"] = metrics["recall"]
                result["f1_score"] = metrics["f1_score"]

            except Exception as e:
                result["error"] = f"Quality evaluation failed: {e}"
                print(f"WARNING: {result['error']}")
    else:
        if config["quality_file"]:
            print(f"WARNING: Quality file not found: {quality_file}")
        else:
            print("INFO: No quality evaluation configured for this query")

    return result


def run_quality_evaluation(quality_file: Path, part_file: str) -> str:
    """Run quality evaluation with the specified part file."""
    # Read the quality file content
    with open(quality_file, 'r') as f:
        quality_code = f.read()

    # Create a modified version that uses our part file
    # Find the function name and call it with our file
    func_name = None
    if "calculate_f1_score" in quality_code:
        func_name = "calculate_f1_score"
    elif "calculate_pair_f1_score" in quality_code:
        func_name = "calculate_pair_f1_score"
    elif "calculate_genre_f1_score" in quality_code:
        func_name = "calculate_genre_f1_score"

    if not func_name:
        return "ERROR: Could not find quality function in file"

    # Create a temporary script that imports and runs the function
    temp_script = f'''
import sys
sys.path.insert(0, "{quality_file.parent}")
exec(open("{quality_file}").read())
{func_name}("{part_file}")
'''

    # Run the temporary script
    process = subprocess.run(
        [sys.executable, "-c", temp_script],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=300,
    )

    return process.stdout + "\n" + process.stderr


def generate_summary(results: list, output_file: Path):
    """Generate a CSV summary of all results."""
    fieldnames = ["query", "status", "latency_seconds", "precision", "recall", "f1_score", "error"]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSummary saved to: {output_file}")


def print_summary_table(results: list):
    """Print a formatted summary table."""
    print("\n" + "="*80)
    print("SEMBENCH SUMMARY REPORT")
    print("="*80)
    print(f"{'Query':<10} {'Status':<12} {'Latency(s)':<12} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-"*80)

    for r in results:
        latency = f"{r['latency_seconds']:.2f}" if r['latency_seconds'] >= 0 else "N/A"
        precision = f"{r['precision']:.4f}" if r['precision'] >= 0 else "N/A"
        recall = f"{r['recall']:.4f}" if r['recall'] >= 0 else "N/A"
        f1 = f"{r['f1_score']:.4f}" if r['f1_score'] >= 0 else "N/A"

        print(f"{r['query']:<10} {r['status']:<12} {latency:<12} {precision:<10} {recall:<10} {f1:<10}")

    print("="*80)

    # Calculate averages for completed queries with valid metrics
    valid_results = [r for r in results if r['status'] == 'completed' and r['f1_score'] >= 0]
    if valid_results:
        avg_latency = sum(r['latency_seconds'] for r in valid_results) / len(valid_results)
        avg_precision = sum(r['precision'] for r in valid_results) / len(valid_results)
        avg_recall = sum(r['recall'] for r in valid_results) / len(valid_results)
        avg_f1 = sum(r['f1_score'] for r in valid_results) / len(valid_results)

        print(f"{'AVERAGE':<10} {'':<12} {avg_latency:<12.2f} {avg_precision:<10.4f} {avg_recall:<10.4f} {avg_f1:<10.4f}")
        print("="*80)


def main():
    """Main execution function."""
    print("="*60)
    print("SemBench Runner")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base Directory: {BASE_DIR}")
    print("="*60)

    # Check if we should run specific queries
    if len(sys.argv) > 1:
        # Filter to only requested queries
        requested = set(sys.argv[1:])
        configs = [c for c in QUERY_CONFIGS if c["name"] in requested]
        if not configs:
            print(f"No matching queries found for: {requested}")
            print(f"Available queries: {[c['name'] for c in QUERY_CONFIGS]}")
            sys.exit(1)
    else:
        configs = QUERY_CONFIGS

    print(f"Queries to run: {[c['name'] for c in configs]}")

    # Run all queries
    results = []
    for config in configs:
        result = run_query(config)
        results.append(result)

    # Print summary table
    print_summary_table(results)

    # Generate summary CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = BASE_DIR / f"sembench_summary_{timestamp}.csv"
    generate_summary(results, summary_file)

    # Also save as latest
    latest_file = BASE_DIR / "sembench_summary_latest.csv"
    generate_summary(results, latest_file)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Return non-zero if any queries failed
    failed = [r for r in results if r['status'] not in ['completed', 'pending']]
    if failed:
        print(f"\nWARNING: {len(failed)} queries had issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
