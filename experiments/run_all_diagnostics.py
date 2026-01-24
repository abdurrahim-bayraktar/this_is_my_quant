"""
Diagnostic Test Runner

Runs all Phase 4 diagnostic tests and generates a comprehensive results document.

Usage:
    python experiments/run_all_diagnostics.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import json
import logging
from datetime import datetime

from config import REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Test configurations
TESTS = [
    {
        "name": "Overfitting Sanity Check",
        "file": "overfit_test.py",
        "description": "Tests if model can memorize synthetic data (no bug in architecture)",
        "timeout_minutes": 5,
    },
    {
        "name": "Signal Quality Analysis",
        "file": "signal_quality_test.py",
        "description": "Measures correlation between sentiment and next-day returns",
        "timeout_minutes": 15,
    },
]


def run_test(test_config: dict, experiments_dir: Path) -> dict:
    """Run a single test and capture output."""
    test_file = experiments_dir / test_config["file"]
    
    if not test_file.exists():
        return {
            "status": "SKIPPED",
            "reason": f"File not found: {test_file}",
            "output": "",
        }
    
    logger.info(f"Running: {test_config['name']}")
    logger.info(f"  File: {test_file}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=test_config["timeout_minutes"] * 60,
            cwd=str(experiments_dir.parent),
        )
        
        return {
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "TIMEOUT",
            "reason": f"Exceeded {test_config['timeout_minutes']} minutes",
            "output": "",
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "reason": str(e),
            "output": "",
        }


def generate_results_document(test_results: dict, output_path: Path):
    """Generate markdown results document."""
    
    lines = [
        "# Diagnostic Test Results",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Test | Status | Description |",
        "|------|--------|-------------|",
    ]
    
    for test_name, result in test_results.items():
        status = result.get("status", "UNKNOWN")
        status_emoji = {"PASS": "✅", "FAIL": "❌", "TIMEOUT": "⏰", "SKIPPED": "⏭️", "ERROR": "💥"}.get(status, "❓")
        desc = result.get("description", "")
        lines.append(f"| {test_name} | {status_emoji} {status} | {desc} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ])
    
    for test_name, result in test_results.items():
        lines.extend([
            f"### {test_name}",
            "",
            f"**Status**: {result.get('status', 'UNKNOWN')}",
            "",
        ])
        
        if result.get("reason"):
            lines.extend([f"**Reason**: {result['reason']}", ""])
        
        stdout = result.get("stdout", "")
        if stdout:
            # Extract key findings from output
            lines.extend([
                "#### Output",
                "",
                "```",
                stdout[-3000:] if len(stdout) > 3000 else stdout,  # Limit output length
                "```",
                "",
            ])
        
        stderr = result.get("stderr", "")
        if stderr and result.get("status") != "PASS":
            lines.extend([
                "#### Errors",
                "",
                "```",
                stderr[-1000:] if len(stderr) > 1000 else stderr,
                "```",
                "",
            ])
        
        lines.append("---")
        lines.append("")
    
    # Write file
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Results document saved to: {output_path}")


def main():
    print("=" * 70)
    print("PHASE 4: DIAGNOSTIC TEST RUNNER")
    print("=" * 70)
    print(f"Running {len(TESTS)} diagnostic tests...")
    print("=" * 70)
    
    experiments_dir = Path(__file__).parent
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPORTS_DIR / f"diagnostics_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all tests
    all_results = {}
    
    for test_config in TESTS:
        test_name = test_config["name"]
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")
        
        result = run_test(test_config, experiments_dir)
        result["description"] = test_config["description"]
        all_results[test_name] = result
        
        print(f"Status: {result['status']}")
        if result.get("stdout"):
            # Print last few lines
            last_lines = result["stdout"].strip().split("\n")[-20:]
            print("\n".join(last_lines))
    
    # Save raw results
    with open(output_dir / "raw_results.json", "w") as f:
        # Filter out large outputs for JSON
        json_results = {}
        for name, res in all_results.items():
            json_results[name] = {
                "status": res.get("status"),
                "description": res.get("description"),
                "return_code": res.get("return_code"),
                "reason": res.get("reason"),
            }
        json.dump(json_results, f, indent=2)
    
    # Generate markdown report
    generate_results_document(all_results, output_dir / "DIAGNOSTIC_RESULTS.md")
    
    # Print summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in all_results.values() if r["status"] == "PASS")
    failed = sum(1 for r in all_results.values() if r["status"] == "FAIL")
    other = len(all_results) - passed - failed
    
    print(f"Passed: {passed}/{len(all_results)}")
    print(f"Failed: {failed}/{len(all_results)}")
    if other > 0:
        print(f"Other: {other}/{len(all_results)}")
    
    print(f"\nFull results: {output_dir / 'DIAGNOSTIC_RESULTS.md'}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    main()
