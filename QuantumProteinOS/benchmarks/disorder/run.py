import argparse
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from qpos.disorder import DisorderNetV6, download_disprot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Use 500-sequence subset')
    parser.add_argument('--full', action='store_true', help='Full evaluation')
    args = parser.parse_args()

    print(f"Running DisorderNet benchmark {'(fast)' if args.fast else '(full)'}...")
    
    # Actually fetch dataset
    df = download_disprot(version="CAID3")
    if args.fast and len(df) > 500:
        df = df.sample(500, random_state=42)
        
    model = DisorderNetV6()
    
    all_y_true = []
    all_y_pred = []
    
    for _, row in df.iterrows():
        seq = row['sequence']
        labels = row['disorder_labels'] # Expected to be 1D array/list of 0s and 1s
        preds = model.predict(seq)
        all_y_true.extend(labels)
        all_y_pred.extend(preds)
        
    # Calculate real AUC
    if len(np.unique(all_y_true)) > 1:
        auc = roc_auc_score(all_y_true, all_y_pred)
    else:
        auc = 0.5 # Default/Mock if no unique targets present
        
    os.makedirs('benchmarks/disorder/results', exist_ok=True)
    with open('benchmarks/disorder/results/metrics.json', 'w') as f:
        json.dump({'auc_roc': auc}, f)
        
if __name__ == '__main__':
    main()
