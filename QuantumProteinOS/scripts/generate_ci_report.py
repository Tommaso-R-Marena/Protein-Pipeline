import argparse
import json
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--commit-sha', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    markdown = f"""# QuantumProteinOS Benchmark Report
**Commit:** {args.commit_sha} | **Run:** {args.run_id} | **Date:** {date_str}

## Results Summary

| Benchmark | Metric | Result | Target | Status |
|-----------|--------|--------|--------|--------|
"""
    
    def gen_row(bench, metric, result, target, passed):
        status = "✅ PASS" if passed else f"❌ FAIL ({result})"
        return f"| {bench} | {metric} | {result} | {target} | {status} |\n"

    disorder_file = os.path.join(args.artifacts_dir, 'disorder-benchmark-results', 'metrics.json')
    if os.path.exists(disorder_file):
        with open(disorder_file) as f:
            d = json.load(f)
            auc = d.get('auc_roc', 0)
            markdown += gen_row("DisorderNet v6", "AUC-ROC", f"{auc:.3f}", "≥ 0.820", auc >= 0.820)
            
    chirality_file = os.path.join(args.artifacts_dir, 'chirality-benchmark-results', 'audit_summary.json')
    if os.path.exists(chirality_file):
        with open(chirality_file) as f:
            d = json.load(f)
            pct = d.get('mean_chirality_pct_correct', 0)
            n_100 = d.get('n_structures_at_100pct', 0)
            markdown += gen_row("ChiralFold", "Mean chirality %", f"{pct:.1f}%", "≥ 96%", pct >= 96.0)
            markdown += gen_row("ChiralFold", "Structures at 100%", f"{n_100}/31", "≥ 28/31", n_100 >= 28)

    rotamer_file = os.path.join(args.artifacts_dir, 'rotamer-benchmark-results', 'comparison_table.json')
    if os.path.exists(rotamer_file):
        with open(rotamer_file) as f:
            d = json.load(f)
            for inst in d.get('instances', []):
                name = inst['name']
                iws = inst['iws_qaoa_gap_pct']
                greedy = inst['greedy_gap_pct']
                markdown += gen_row(f"IWS-QAOA {name}", "Gap vs exact", f"{iws:.1f}%", f"≤ greedy {greedy:.1f}%", iws <= greedy)
                
    autoinhib_file = os.path.join(args.artifacts_dir, 'autoinhibited-benchmark-results', 'stats.json')
    if os.path.exists(autoinhib_file):
        with open(autoinhib_file) as f:
            d = json.load(f)
            qicess = d.get('qicess_frac_below_3A', 0)
            af3 = d.get('af3_frac_below_3A', 1)
            wilcox = d.get('wilcoxon_p_qicess_vs_af3', 1)
            markdown += gen_row("QICESS", "imfdRMSD < 3Å", f"{qicess*100:.0f}%", f"> AF3 {af3*100:.0f}%", qicess > af3)
            markdown += gen_row("QICESS", "Wilcoxon p", f"{wilcox:.4f}", "< 0.05", wilcox < 0.05)

    smoke_test_dir = os.path.join(args.artifacts_dir, 'pipeline-smoke-test-outputs')
    if os.path.exists(smoke_test_dir):
        files_found = 0
        required = ['structure.pdb', 'disorder_scores.csv', 'chirality_report.json', 'qadf_score.json', 'summary.txt']
        for req in required:
            if os.path.exists(os.path.join(smoke_test_dir, req)):
                files_found += 1
        if os.path.exists(os.path.join(smoke_test_dir, 'ensemble')):
            files_found += 1
            
        markdown += gen_row("Pipeline", "Output completeness", f"{files_found}/6 files", "6/6 files", files_found == 6)

    markdown += """
## Simulation Disclaimer
⚠️ All quantum results are CLASSICALLY SIMULATED via PennyLane default.qubit.
No QPU hardware was used. No quantum advantage is claimed at ≤25 qubits.

## Honest Limitations
- VQE does not outperform random ranking at 16 qubits (p=0.25)
- QICESS benchmark uses calibrated simulations, not real AF3 inference
- DisorderNet fast mode uses 500-sequence DisProt subset
"""
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(markdown)
        
if __name__ == '__main__':
    main()
