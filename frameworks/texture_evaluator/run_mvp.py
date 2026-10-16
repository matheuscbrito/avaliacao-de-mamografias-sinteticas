"""Executa GLCM + Wavelet e salva prévias para inspeção visual."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from wavelet.src.load_data import CasePaths, discover_cases
from types import SimpleNamespace
from wavelet.src.validate_cases import ValidCase, validate_case
from wavelet.src.density_features import density_features, density_limits, density_shift, density_status, reference_interval
from wavelet.src.densidade_wavelet import wavelet_verdict
from wavelet.src.wavelet_diagnostico import raw_readings, relative
from wavelet.src.wavelet_features_fullimage import wavelet_energy_entropy_fullimage
from wavelet.src.wavelet_brightness import ll_mean
from frameworks.texture_evaluator.nex_referencia import measure as nex_measure, summary as nex_summary
from frameworks.texture_evaluator.glcm.src import RobustScale, calibrate_robust_scale, extract_glcm
from frameworks.texture_evaluator.decision import GLCM_METRICS, WAVELET_METRICS, decide, glcm_metric_columns, wavelet_metric_columns
from frameworks.texture_evaluator.regional import apply_pair_check, calibrate_maps, pair_correlation, regional, regional_columns
from frameworks.texture_evaluator.report import generate_report

def calibrate(valid) -> SimpleNamespace:
    """Réguas e limites tirados só das originais válidas; valem para todos os pares."""
    scale = calibrate_robust_scale(np.concatenate([c.original[c.lung_mask] for c in valid]), tukey_k=3, levels=32)
    interval = reference_interval([(c.original, c.lung_mask) for c in valid])
    # eixo de densidade da Wavelet: segunda métrica de brilho, lado a lado com Δμ/Sat31 (convivência ainda em aberto)
    limits = density_limits([density_features(c.original, c.lung_mask, interval) for c in valid])
    # régua única: a Wavelet corta e normaliza com a mesma régua Tukey k=3 da GLCM (ver teste_regua.py);
    # a densidade continua na faixa P1–P99, que é a definição de "fora da faixa real"
    ruler = {'p1': scale.clip_low, 'p99': scale.clip_high}
    means = np.array([c.original[c.lung_mask].mean() for c in valid]); mean_iqr = float(np.diff(np.percentile(means, [25, 75]))[0])
    # NEx é medido nas mesmas janelas inteiramente dentro da ROI pulmonar. A
    # faixa P5–P95 nasce somente das originais e fica congelada no histórico.
    nex_reference = nex_summary(np.array([nex_measure(c.original, c.lung_mask)["nex_roi"] for c in valid]))
    settings = SimpleNamespace(wavelet='haar', min_tissue=.9)
    # IQR de cada métrica Wavelet entre as originais: a régua da classificação por métrica (decision.wavelet_metric_columns)
    readings = [raw_readings(c.original, c.lung_mask, ruler, settings) for c in valid]
    wavelet_iqr = {key: float(np.subtract(*np.percentile([r[key] for r in readings], (75, 25)))) for key in WAVELET_METRICS}
    # o mesmo para a GLCM (contraste e energia), na régua Tukey k=3
    glcm = [glcm_reading(extract_glcm(c.original, c.lung_mask, scale)) for c in valid]
    glcm_iqr = {key: float(np.subtract(*np.percentile([g[key] for g in glcm], (75, 25)))) for key in GLCM_METRICS}
    return SimpleNamespace(scale=scale, interval=interval, limits=limits, ruler=ruler, mean_iqr=mean_iqr, settings=settings,
                           maps=calibrate_maps(valid, ruler, interval, limits), nex_reference=nex_reference, wavelet_iqr=wavelet_iqr, glcm_iqr=glcm_iqr)

def glcm_reading(features: dict) -> dict:
    return {key: features[f'glcm_{key}'] for key in GLCM_METRICS}

def context_to_dict(ctx) -> dict:
    """Réguas e limites em JSON, para reaplicar a mesma calibração mais tarde."""
    return {'glcm_scale': ctx.scale.as_dict(), 'density_interval': ctx.interval, 'density_limits': ctx.limits, 'mean_iqr': ctx.mean_iqr, 'maps': ctx.maps, 'nex_reference': ctx.nex_reference, 'wavelet_iqr': ctx.wavelet_iqr, 'glcm_iqr': ctx.glcm_iqr}

def context_from_dict(data: dict) -> SimpleNamespace:
    scale = RobustScale(**data['glcm_scale'])
    return SimpleNamespace(scale=scale, interval=data['density_interval'], limits=data['density_limits'], ruler={'p1': scale.clip_low, 'p99': scale.clip_high},
                           mean_iqr=data['mean_iqr'], settings=SimpleNamespace(wavelet='haar', min_tissue=.9), maps=data['maps'], nex_reference=data.get('nex_reference'), wavelet_iqr=data.get('wavelet_iqr'), glcm_iqr=data.get('glcm_iqr'))

def evaluate_pair(case_id, original, synthetic, mask, ctx) -> dict:
    """Mede brilho, GLCM e Wavelet de um par e devolve a linha com a decisão final na frente."""
    scale, interval, limits, ruler, settings = ctx.scale, ctx.interval, ctx.limits, ctx.ruler, ctx.settings
    real_g, syn_g = extract_glcm(original,mask,scale), extract_glcm(synthetic,mask,scale)
    real_w=wavelet_energy_entropy_fullimage(original,mask,ruler['p1'],ruler['p99'],'haar',2)
    syn_w=wavelet_energy_entropy_fullimage(synthetic,mask,ruler['p1'],ruler['p99'],'haar',2)
    delta=float(synthetic[mask].mean()-original[mask].mean()); sat=syn_g['saturation_tone_31_pct']
    dens_real, dens_syn = density_features(original,mask,interval), density_features(synthetic,mask,interval)
    nex_real, nex_synthetic = nex_measure(original, mask), nex_measure(synthetic, mask)
    if nex_real['poucas_janelas'] or nex_synthetic['poucas_janelas']:
        nex_status = 'inconclusivo'
    elif nex_synthetic['nex_roi'] > ctx.nex_reference['p95']:
        nex_status = 'alto'
    elif nex_synthetic['nex_roi'] < ctx.nex_reference['p5']:
        nex_status = 'baixo'
    else:
        nex_status = 'ok'
    # veredito da Wavelet: energia fina/grossa da sintética em % da original (limites exploratórios)
    raw_real, raw_synthetic = raw_readings(original,mask,ruler,settings), raw_readings(synthetic,mask,ruler,settings)
    reading = relative(raw_synthetic, raw_real)
    status='hard_reject' if abs(delta)>2*ctx.mean_iqr or sat>15 else 'manual_review' if abs(delta)>ctx.mean_iqr else 'approved'
    row={'case_id':case_id,'brightness_status':status,'delta_mu':delta,'sat31_pct':sat,
         'density_status':density_status(dens_syn,dens_real,limits),'density_median_shift_iqr':density_shift(dens_syn,dens_real,limits)['median_shift_iqr'],
         'wavelet_verdict':wavelet_verdict(reading),'wavelet_fine_pct':reading['fine'],'wavelet_coarse_pct':reading['coarse'],'wavelet_ratio':reading['ratio'],
         'glcm_contrast_real':real_g['glcm_contrast'],'glcm_contrast_synthetic':syn_g['glcm_contrast'],
         'glcm_energy_real':real_g['glcm_energy'],'glcm_energy_synthetic':syn_g['glcm_energy'],
         'wavelet_energy_real':real_w['energy'],'wavelet_energy_synthetic':syn_w['energy'],
         'wavelet_entropy_real':real_w['entropy'],'wavelet_entropy_synthetic':syn_w['entropy'],
         'wavelet_ll_real':ll_mean(original,mask,ruler['p1'],ruler['p99'],'haar',2,.9),
         'wavelet_ll_synthetic':ll_mean(synthetic,mask,ruler['p1'],ruler['p99'],'haar',2,.9),
         'nex_real':nex_real['nex_roi'],'nex_synthetic':nex_synthetic['nex_roi'],
         'nex_delta':nex_synthetic['nex_roi'] - nex_real['nex_roi'], 'nex_status':nex_status,
         'nex_janelas_real':nex_real['janelas'],'nex_janelas_sintetica':nex_synthetic['janelas'],
         **(wavelet_metric_columns(raw_real, raw_synthetic, ctx.wavelet_iqr) if ctx.wavelet_iqr else {}),
         **(glcm_metric_columns(glcm_reading(real_g), glcm_reading(syn_g), ctx.glcm_iqr) if ctx.glcm_iqr else {}),
         'pair_correlation':pair_correlation(original,synthetic,mask), **regional_columns(regional(original,synthetic,mask,ctx)[2])}
    # decisão final na frente da tabela: veredito, motivos e depois as medidas
    return {'case_id': case_id, **decide(row), **row}

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True, help="Pasta com original e _generate")
    p.add_argument("--mask-dir", type=Path, help="Pasta das máscaras; obrigatória para Dsr")
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--preview-limit", type=int, default=-1)
    a = p.parse_args(); valid = []
    for paths in discover_cases(a.data_dir):
        if a.mask_dir:
            paths = CasePaths(paths.case_id, paths.original, paths.synthetic, a.mask_dir / f"{paths.case_id}_mask.tiff")
        result = validate_case(paths, erosion_radius=2)
        if isinstance(result, ValidCase): valid.append(result)
    if not valid: raise RuntimeError("nenhum par válido")
    ctx = calibrate(valid); scale, interval, limits = ctx.scale, ctx.interval, ctx.limits
    rows=[]; previews=a.output_dir/"previews"; previews.mkdir(parents=True, exist_ok=True)
    for n, c in enumerate(valid):
        rows.append(evaluate_pair(c.paths.case_id, c.original, c.synthetic, c.lung_mask, ctx))
    apply_pair_check(rows)
    for n, (c, row) in enumerate(zip(valid, rows)):
        status, delta, sat = row['brightness_status'], row['delta_mu'], row['sat31_pct']
        if a.preview_limit < 0 or n<a.preview_limit:
            fig,ax=plt.subplots(1,3,figsize=(12,4))
            for axis,img,title in zip(ax[:2],(c.original,c.synthetic),('Real','Sintética')):
                axis.imshow(img,cmap='gray',vmin=interval['p1'],vmax=interval['p99']); axis.contour(c.lung_mask,[.5],colors='#12a57a'); axis.set_title(title); axis.axis('off')
            ax[2].imshow(c.lung_mask,cmap='gray'); ax[2].set_title('ROI erodida'); ax[2].axis('off')
            fig.suptitle(f"{c.paths.case_id} | final={row['final_status']} | brilho={status} | Δμ={delta:.3f} | Sat31={sat:.1f}% | densidade={row['density_status']} | wavelet={row['wavelet_verdict']}",fontsize=9)
            fig.tight_layout(); fig.savefig(previews/f'{n:02d}.png',dpi=150); plt.close(fig)
    with (a.output_dir/'mvp_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (a.output_dir/'mvp_protocol.json').write_text(json.dumps({'valid_pairs':len(valid),'glcm_scale':scale.as_dict(),'texture_ruler':'glcm_scale (Tukey k=3), compartilhada por GLCM e Wavelet','density_interval':interval,'density_limits':limits,'third_metric':'reserved'},indent=2),encoding='utf-8')
    generate_report(rows, a.output_dir)
if __name__=='__main__': main()
