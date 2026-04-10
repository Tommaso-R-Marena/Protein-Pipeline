import os
import json
from qpos.chirality.auditor import ChiralFoldAuditor
from qpos.data.pdb_parser import PDBParser

def main():
    print("Running ChiralFold validation...")
    
    auditor = ChiralFoldAuditor()
    
    # Target our local testing fixtures
    fixture_dir = "tests/fixtures"
    fixture_files = [os.path.join(fixture_dir, f) for f in os.listdir(fixture_dir) if f.endswith('.pdb')]
    
    correct_pcts = []
    perfect_structs = 0
    
    for f in fixture_files:
        parser = PDBParser(f)
        report = auditor.audit(parser.structure)
        pct = report['chirality']['pct_correct']
        correct_pcts.append(pct)
        if pct == 100.0:
            perfect_structs += 1
            
    if len(correct_pcts) > 0:
        mean_pct = sum(correct_pcts) / len(correct_pcts)
    else:
        mean_pct = 0.0
        
    os.makedirs('benchmarks/chirality/results', exist_ok=True)
    with open('benchmarks/chirality/results/audit_summary.json', 'w') as f:
        json.dump({
            'mean_chirality_pct_correct': mean_pct, 
            'n_structures_at_100pct': perfect_structs
        }, f)

if __name__ == '__main__':
    main()
