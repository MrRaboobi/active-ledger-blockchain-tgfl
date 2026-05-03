import numpy as np
base = r'C:\Users\T14s\Desktop\FYP-Blockchain-FL\friends experimentation'

print('=== GAUSSIAN (Robust Results) per-round mean F1 history ===')
d = np.load(base+r'\robust_comparison (1).npy', allow_pickle=True).item()
for m in ['G_PoC_Only','C_MultiKrum','E_TrimmedMean','F_Bulyan','D_Median','B_Krum','A_FedAvg']:
    r = d[m]['round_f1']
    print(f'  {m}: R1={r[0]:.4f} R10={r[9]:.4f} R20={r[19]:.4f} R30={r[29]:.4f} R40={r[-1]:.4f} MAX={r.max():.4f}')

print('\n=== GAUSSIAN Final precision/recall/support ===')
for m in ['G_PoC_Only','A_FedAvg','B_Krum','C_MultiKrum','D_Median','E_TrimmedMean','F_Bulyan']:
    v=d[m]
    print(f'  {m}')
    print(f'    prec    = {v["final_precision"]}')
    print(f'    recall  = {v["final_recall"]}')
    print(f'    f1      = {v["final_f1"]}')
    print(f'    support = {v["final_support"]}')
    print(f'    elapsed_h = {v["elapsed_h"]:.3f}')

print('\n=== SEMANTIC (label-flip) per-round and final ===')
d = np.load(base+r'\semantic_results_recovered.npy', allow_pickle=True).item()
for m,v in d.items():
    r=v['round_f1']
    print(f'  {m}: round_f1 len={len(r)} R1={r[0]:.4f} R15={r[14]:.4f} R40={r[-1]:.4f} MAX={r.max():.4f} final={v["final_f1"]} lat_mean={np.mean(v["latency_history"]):.2f}')

print('\n=== SLEEPER per-round and final ===')
d = np.load(base+r'\sota_sleeper_results.npy', allow_pickle=True).item()
for m,v in d.items():
    r=v['round_f1']
    print(f'  {m}: R1={r[0]:.4f} R10={r[9]:.4f} R14={r[13]:.4f} R15={r[14]:.4f} R16={r[15]:.4f} R20={r[19]:.4f} R30={r[29]:.4f} R40={r[-1]:.4f} MAX={r.max():.4f}')
    print(f'     final_f1 = {v["final_f1"]}')

print('\n=== SESSION 4 (Trust-Gated LDM vs MultiKrum BlindLDM) ===')
d = np.load(base+r'\session4_results.npy', allow_pickle=True).item()
for m,v in d.items():
    print(f'--- {m} ---')
    print(f'   round_f1:     R1={v["round_f1"][0]:.4f} R10={v["round_f1"][9]:.4f} R14={v["round_f1"][13]:.4f} R15={v["round_f1"][14]:.4f} R20={v["round_f1"][19]:.4f} R30={v["round_f1"][29]:.4f} R40={v["round_f1"][-1]:.4f} MAX={v["round_f1"].max():.4f}')
    print(f'   round_apb_f1: R1={v["round_apb_f1"][0]:.4f} R10={v["round_apb_f1"][9]:.4f} R14={v["round_apb_f1"][13]:.4f} R15={v["round_apb_f1"][14]:.4f} R20={v["round_apb_f1"][19]:.4f} R30={v["round_apb_f1"][29]:.4f} R40={v["round_apb_f1"][-1]:.4f} MAX={v["round_apb_f1"].max():.4f}')
    print(f'   round_bsr:    R1={v["round_bsr"][0]:.4f} R10={v["round_bsr"][9]:.4f} R15={v["round_bsr"][14]:.4f} R20={v["round_bsr"][19]:.4f} R30={v["round_bsr"][29]:.4f} R40={v["round_bsr"][-1]:.4f} MEAN={v["round_bsr"].mean():.4f} MIN={v["round_bsr"].min():.4f} MAX={v["round_bsr"].max():.4f}')
    print(f'   final_f1 = {v["final_f1"]}')
    print(f'   avg latency = {v["round_lat"].mean():.4f}s')
