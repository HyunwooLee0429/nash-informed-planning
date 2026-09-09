"""Driver: EBMC Nash-informed planning on the Minnesota AIS dataset.

Writes canonical result files to nash_informed/ebmc/results/minnesota/.
Re-running this script regenerates (and overwrites) those files.

Output files per --percentage:
    df_test_BZR_MN_{pct}.csv
    df_test_minnesota_SW_enhancing_{pct}.csv
    df_details_BZR_MN_{pct}.csv
    df_details_minnesota_SW_enhancing_{pct}.csv
    all_PNE_sets_{pct}.pkl
    all_selected_lakes_{pct}.pkl
    coalitions_{pct}.csv
With --include-tailored:
    df_test_MN_tailored.csv, df_MN_details_tailored.csv,
    coalitions_tailored.csv, all_selected_lakes_tailored.pkl
"""

from __future__ import annotations

import argparse, pickle, sys, time
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from nash_informed.ebmc import (
    load_minnesota_instance,
    solve_sw, solve_gc, solve_bgc, solve_bsw, solve_bc, solve_non_game,
    find_pne_bzr, compute_utility,
    bc1_kwargs, bc2_kwargs, bc3_kwargs, nvc_kwargs,
)
from nash_informed.common import construct_reference_utility


DEFAULT_OUT = REPO_ROOT / "nash_informed" / "ebmc" / "results" / "minnesota"


def _u_per_county(inst, selected, counties):
    u, _ = compute_utility(inst, selected, only_playing=True)
    return {c: u.get(c, 0.0) for c in counties}


def run_single_percentage(pct, args, cfg, out_dir):
    print(f"\n{'='*80}\n[{pct}] loading Minnesota instance ...\n{'='*80}")
    inst = load_minnesota_instance(data_dir=args.data_dir, percentage=pct)
    print(f"  {inst.summary()}")
    counties_playing = inst.counties_playing
    n_cts = inst.n_counties
    n_cts_playing = inst.n_counties_playing
    n_lakes = len(inst.lakes)
    n_infested_playing = sum(1 for lake in inst.infested_lakes
                              if inst.lake_county.get(lake) in counties_playing)
    total_budget = inst.total_budget

    print(f"\n[{pct}] Solving SW ...")
    sw = solve_sw(inst, time_limit=args.bzr_time_limit, only_playing=True, config=cfg)
    print(f"  SW phi = {sw['phi']:.0f}  runtime={sw['wall_seconds']:.1f}s")

    print(f"\n[{pct}] Running BZR max ...")
    pne_max, info_max = find_pne_bzr(inst, direction='max',
                                       time_limit=args.bzr_time_limit,
                                       only_playing=True, config=cfg, verbose=True)
    print(f"\n[{pct}] Running BZR min ...")
    pne_min, info_min = find_pne_bzr(inst, direction='min',
                                       time_limit=args.bzr_time_limit,
                                       only_playing=True, config=cfg, verbose=True)

    seen, pne_list = set(), []
    for r in pne_max + pne_min:
        key = tuple(sorted(r['selected_lakes']))
        if key not in seen:
            seen.add(key); pne_list.append(r)
    best_pne = max(pne_list, key=lambda d: d['phi']) if pne_list else None
    worst_pne = min(pne_list, key=lambda d: d['phi']) if pne_list else None
    bpne_phi = best_pne['phi'] if best_pne else float('nan')
    wpne_phi = worst_pne['phi'] if worst_pne else float('nan')
    rpne_phi = bpne_phi

    simple = [{'u_player': r['u_player'], 'phi': r['phi']} for r in pne_list]
    ref_u, ref_mode, _, _ = construct_reference_utility(
        simple, counties_playing, scheme='dominant_or_average')

    print(f"\n[{pct}] Solving BSW ...")
    bsw = solve_bsw(inst, ir_ref=ref_u, time_limit=args.bsw_time_limit,
                     only_playing=True,
                     warm_start=best_pne['selected_lakes'] if best_pne else None,
                     config=cfg)
    if bsw is None:
        ref_u2, _, _, _ = construct_reference_utility(simple, counties_playing,
                                                       scheme='min_pne')
        bsw = solve_bsw(inst, ir_ref=ref_u2, time_limit=args.bsw_time_limit,
                         only_playing=True,
                         warm_start=best_pne['selected_lakes'] if best_pne else None,
                         config=cfg)
    print(f"  BSW phi = {bsw['phi']:.0f}  runtime={bsw['wall_seconds']:.1f}s")

    print(f"\n[{pct}] Solving BGC ...")
    bgc = solve_bgc(inst, ir_ref=ref_u, time_limit=args.bsw_time_limit,
                     only_playing=True,
                     warm_start=best_pne['selected_lakes'] if best_pne else None,
                     config=cfg)
    print(f"  BGC phi = {bgc['phi']:.0f}  runtime={bgc['wall_seconds']:.1f}s")

    print(f"\n[{pct}] Solving BC ...")
    bc = solve_bc(inst, ir_ref=ref_u, time_limit=args.bc_time_limit,
                   only_playing=True,
                   warm_start=best_pne['selected_lakes'] if best_pne else None,
                   config=cfg)
    print(f"  BC phi = {bc['phi']:.0f}  runtime={bc['wall_seconds']:.1f}s")

    print(f"\n[{pct}] Solving GC ...")
    gc = solve_gc(inst, time_limit=args.bsw_time_limit, only_playing=True, config=cfg)
    print(f"  GC phi = {gc['phi']:.0f}  runtime={gc['wall_seconds']:.1f}s")

    if args.include_nsg:
        print(f"\n[{pct}] Solving non-game baselines (per-county) ...")
        nsg_per = {}
        for c in counties_playing:
            u, _ = solve_non_game(inst, c, config=cfg)
            nsg_per[c] = u
        nsg_total = sum(nsg_per.values())
        nsg_nc_per = dict(nsg_per)
        nsg_nc_total = nsg_total
    else:
        nsg_total = float('nan'); nsg_nc_total = float('nan')
        nsg_per = {c: float('nan') for c in counties_playing}
        nsg_nc_per = dict(nsg_per)

    out_dir.mkdir(parents=True, exist_ok=True)

    df_bzr_test = pd.DataFrame({
        'num_cts': [n_cts], 'num_playing_cts': [n_cts_playing],
        'num_lakes': [n_lakes], 'num_infested_lakes': [n_infested_playing],
        'budget_ratio': [''], 'budget': [total_budget],
        'BZR_min_1st': [info_min.get('first_PNE_obj')],
        'BZR_min': [wpne_phi],
        'BZR_max_1st': [info_max.get('first_PNE_obj')],
        'BZR_max': [bpne_phi],
        'MIPGAP_BZR_min': [info_min.get('mipgap')],
        'MIPGAP_BZR_max': [info_max.get('mipgap')],
        'OSW': [sw['phi']],
        'Cuts_BZR_min': [info_min.get('EI_cuts')],
        'additional_Cuts_BZR_min': [info_min.get('additional_cuts')],
        'Cuts_BZR_max': [info_max.get('EI_cuts')],
        'additional_Cuts_BZR_max': [info_max.get('additional_cuts')],
        'BZR_min_1st_T': [info_min.get('first_time')],
        'BZR_min_T': [info_min.get('runtime')],
        'BZR_max_1st_T': [info_max.get('first_time')],
        'BZR_max_T': [info_max.get('runtime')],
        'num_PNEs_BZR_min': [len(pne_min)],
        'num_PNEs_BZR_max': [len(pne_max)],
        'num_unique_PNEs': [len(pne_list)],
        'is_individually_best': [ref_mode == 'dominant'],
        'Status_BZR_max': [info_max.get('status')],
        'Status_BZR_min': [info_min.get('status')],
    }, index=['Total'])
    df_bzr_test.to_csv(out_dir / f'df_test_BZR_MN_{pct}.csv')

    df_sw_test = pd.DataFrame({
        'num_cts': [n_cts], 'num_cts_playing': [n_cts_playing],
        'num_lakes': [n_lakes], 'num_infested_lakes': [n_infested_playing],
        'budget': [total_budget],
        'NSG-NC': [nsg_nc_total], 'NSG': [nsg_total],
        'WPNE': [wpne_phi], 'BPNE': [bpne_phi], 'RPNE': [rpne_phi],
        'BSW': [bsw['phi']], 'SW': [sw['phi']],
        'BC': [bc['phi']], 'BGC': [bgc['phi']], 'GC': [gc['phi']],
        'NSG-NC_T': [float('nan')], 'NSG_T': [float('nan')],
        'BSW_T': [bsw['wall_seconds']], 'SW_T': [sw['wall_seconds']],
        'BC_T': [bc['wall_seconds']], 'BGC_T': [bgc['wall_seconds']],
        'GC_T': [gc['wall_seconds']],
        'BSW_MG': [bsw['mipgap']], 'SW_MG': [sw['mipgap']],
        'BC_MG': [bc['mipgap']], 'BGC_MG': [bgc['mipgap']],
        'GC_MG': [gc['mipgap']],
    }, index=['Total'])
    df_sw_test.to_csv(out_dir / f'df_test_minnesota_SW_enhancing_{pct}.csv')

    df_bzr_det = pd.DataFrame({
        'budget': [inst.county_budget[c] for c in counties_playing],
        'BZR_min': [worst_pne['u_player'].get(c, 0.0) if worst_pne else 0.0
                     for c in counties_playing],
        'BZR_max': [best_pne['u_player'].get(c, 0.0) if best_pne else 0.0
                     for c in counties_playing],
    }, index=counties_playing)
    df_bzr_det.to_csv(out_dir / f'df_details_BZR_MN_{pct}.csv')

    u_sw  = _u_per_county(inst, sw['selected_lakes'],  counties_playing)
    u_bsw = _u_per_county(inst, bsw['selected_lakes'], counties_playing)
    u_bc  = _u_per_county(inst, bc['selected_lakes'],  counties_playing)
    u_bgc = _u_per_county(inst, bgc['selected_lakes'], counties_playing)
    u_gc  = _u_per_county(inst, gc['selected_lakes'],  counties_playing)
    inf_per = {c: sum(1 for lake in inst.infested_lakes
                       if inst.lake_county.get(lake) == c)
                for c in counties_playing}
    df_sw_det = pd.DataFrame({
        'infested_lakes': [inf_per[c] for c in counties_playing],
        'budget': [inst.county_budget[c] for c in counties_playing],
        'NSG-NC': [nsg_nc_per[c] for c in counties_playing],
        'NSG':    [nsg_per[c]    for c in counties_playing],
        'WPNE':   [worst_pne['u_player'].get(c, 0.0) if worst_pne else 0.0
                    for c in counties_playing],
        'BPNE':   [best_pne['u_player'].get(c, 0.0)  if best_pne  else 0.0
                    for c in counties_playing],
        'RPNE':   [best_pne['u_player'].get(c, 0.0)  if best_pne  else 0.0
                    for c in counties_playing],
        'BSW': [u_bsw[c] for c in counties_playing],
        'SW':  [u_sw[c]  for c in counties_playing],
        'BC':  [u_bc[c]  for c in counties_playing],
        'BGC': [u_bgc[c] for c in counties_playing],
        'GC':  [u_gc[c]  for c in counties_playing],
    }, index=counties_playing)
    df_sw_det.to_csv(out_dir / f'df_details_minnesota_SW_enhancing_{pct}.csv')

    with open(out_dir / f'all_PNE_sets_{pct}.pkl', 'wb') as f:
        pickle.dump({'BPNE': pne_max, 'WPNE': pne_min, 'deduped': pne_list}, f)
    with open(out_dir / f'all_selected_lakes_{pct}.pkl', 'wb') as f:
        pickle.dump({
            'SW':  sw['selected_lakes'], 'BSW': bsw['selected_lakes'],
            'BC':  bc['selected_lakes'], 'BGC': bgc['selected_lakes'],
            'GC':  gc['selected_lakes'],
            'BPNE': best_pne['selected_lakes'] if best_pne else [],
            'WPNE': worst_pne['selected_lakes'] if worst_pne else [],
        }, f)

    if bc.get('coalition_assignment'):
        ca = bc['coalition_assignment']
        pd.DataFrame({'county': list(ca.keys()),
                      'group':  list(ca.values())}).to_csv(
            out_dir / f'coalitions_{pct}.csv', index=False)

    if args.include_tailored:
        print(f"\n[{pct}] Solving tailored BC variants ...")
        tailored = {'BC_0': bc}
        tailored['BC_1'] = solve_bc(
            inst, ir_ref=ref_u, time_limit=args.bc_time_limit, only_playing=True,
            warm_start=best_pne['selected_lakes'] if best_pne else None,
            config=cfg, **bc1_kwargs(counties_playing))
        tailored['BC_2'] = solve_bc(
            inst, ir_ref=ref_u, time_limit=args.bc_time_limit, only_playing=True,
            warm_start=best_pne['selected_lakes'] if best_pne else None,
            config=cfg, **bc2_kwargs(counties_playing))
        tailored['BC_3'] = solve_bc(
            inst, ir_ref=ref_u, time_limit=args.bc_time_limit, only_playing=True,
            warm_start=best_pne['selected_lakes'] if best_pne else None,
            config=cfg, **bc3_kwargs(counties_playing))
        nvk = nvc_kwargs(counties_playing)
        tailored['NVC'] = solve_bc(
            inst, ir_ref={c: 0.0 for c in counties_playing},
            time_limit=args.bc_time_limit, only_playing=True, config=cfg, **nvk)
        tailored['BC_NVC'] = solve_bc(
            inst, ir_ref=ref_u, time_limit=args.bc_time_limit, only_playing=True,
            warm_start=best_pne['selected_lakes'] if best_pne else None,
            config=cfg, **nvk)

        df_tailored = pd.DataFrame({
            'num_cts': [n_cts], 'num_lakes': [n_lakes],
            'num_infested_lakes': [n_infested_playing], 'budget': [total_budget],
            **{name: [r['phi']] for name, r in tailored.items()},
            **{f'{name}_T': [r['wall_seconds']] for name, r in tailored.items()},
            **{f'{name}_MG': [r['mipgap']] for name, r in tailored.items()},
        }, index=['Total'])
        df_tailored.to_csv(out_dir / 'df_test_MN_tailored.csv')

        rows = {}
        for c in counties_playing:
            row = {'budget': inst.county_budget[c]}
            for name, r in tailored.items():
                u = _u_per_county(inst, r['selected_lakes'], counties_playing)
                row[name] = u[c]
            rows[c] = row
        pd.DataFrame.from_dict(rows, orient='index').to_csv(
            out_dir / 'df_MN_details_tailored.csv')

        if tailored['BC_3'].get('coalition_assignment'):
            ca = tailored['BC_3']['coalition_assignment']
            pd.DataFrame({'county': list(ca.keys()),
                          'group': list(ca.values())}).to_csv(
                out_dir / 'coalitions_tailored.csv', index=False)

        sel_tailored = {n: r['selected_lakes'] for n, r in tailored.items()}
        with open(out_dir / 'all_selected_lakes_tailored.pkl', 'wb') as f:
            pickle.dump(sel_tailored, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=str(REPO_ROOT / 'data'))
    parser.add_argument('--percentage', nargs='+', default=['100%'])
    parser.add_argument('--bzr-time-limit', type=int, default=3600)
    parser.add_argument('--bsw-time-limit', type=int, default=3600)
    parser.add_argument('--bc-time-limit',  type=int, default=7200)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--include-nsg', action='store_true')
    parser.add_argument('--include-tailored', action='store_true')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUT))
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    cfg = {'gurobi_threads': args.threads}
    out_dir = Path(args.output_dir)
    start = time.time()
    for pct in args.percentage:
        run_single_percentage(pct, args, cfg, out_dir)
    print(f'\nAll done in {(time.time()-start)/60:.1f} min. Outputs under: {out_dir}')


if __name__ == '__main__':
    main()
