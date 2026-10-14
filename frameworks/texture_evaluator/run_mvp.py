"""Executa GLCM + Wavelet e salva prévias para inspeção visual."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from wavelet.src.load_data import CasePaths, discover_cases
from wavelet.src.validate_cases import ValidCase, validate_case
from wavelet.src.density_features import reference_interval
from wavelet.src.wavelet_features_fullimage import wavelet_energy_entropy_fullimage
from wavelet.src.wavelet_brightness import ll_mean
from frameworks.texture_evaluator.glcm.src import calibrate_robust_scale, extract_glcm

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True, help="Pasta com original e _generate")
    p.add_argument("--mask-dir", type=Path, help="Pasta das máscaras; obrigatória para Dsr")
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--preview-limit", type=int, default=12)
    a = p.parse_args(); valid = []
    for paths in discover_cases(a.data_dir):
        if a.mask_dir:
            paths = CasePaths(paths.case_id, paths.original, paths.synthetic, a.mask_dir / f"{paths.case_id}_mask.tiff")
        result = validate_case(paths, erosion_radius=2)
        if isinstance(result, ValidCase): valid.append(result)
    if not valid: raise RuntimeError("nenhum par válido")
    scale = calibrate_robust_scale(np.concatenate([c.original[c.lung_mask] for c in valid]), tukey_k=3, levels=32)
    interval = reference_interval([(c.original, c.lung_mask) for c in valid])
    means = np.array([c.original[c.lung_mask].mean() for c in valid]); mean_iqr = float(np.diff(np.percentile(means, [25, 75]))[0])
    rows=[]; previews=a.output_dir/"previews"; previews.mkdir(parents=True, exist_ok=True)
    for n, c in enumerate(valid):
        real_g, syn_g = extract_glcm(c.original,c.lung_mask,scale), extract_glcm(c.synthetic,c.lung_mask,scale)
        real_w=wavelet_energy_entropy_fullimage(c.original,c.lung_mask,interval['p1'],interval['p99'],'haar',2)
        syn_w=wavelet_energy_entropy_fullimage(c.synthetic,c.lung_mask,interval['p1'],interval['p99'],'haar',2)
        delta=float(c.synthetic[c.lung_mask].mean()-c.original[c.lung_mask].mean()); sat=syn_g['saturation_tone_31_pct']
        status='hard_reject' if abs(delta)>2*mean_iqr or sat>15 else 'manual_review' if abs(delta)>mean_iqr else 'approved'
        row={'case_id':c.paths.case_id,'brightness_status':status,'delta_mu':delta,'sat31_pct':sat,
             'glcm_contrast_real':real_g['glcm_contrast'],'glcm_contrast_synthetic':syn_g['glcm_contrast'],
             'glcm_energy_real':real_g['glcm_energy'],'glcm_energy_synthetic':syn_g['glcm_energy'],
             'wavelet_energy_real':real_w['energy'],'wavelet_energy_synthetic':syn_w['energy'],
             'wavelet_entropy_real':real_w['entropy'],'wavelet_entropy_synthetic':syn_w['entropy'],
             'wavelet_ll_real':ll_mean(c.original,c.lung_mask,interval['p1'],interval['p99'],'haar',2,.9),
             'wavelet_ll_synthetic':ll_mean(c.synthetic,c.lung_mask,interval['p1'],interval['p99'],'haar',2,.9)}
        rows.append(row)
        if n<a.preview_limit:
            fig,ax=plt.subplots(1,3,figsize=(12,4))
            for axis,img,title in zip(ax[:2],(c.original,c.synthetic),('Real','Sintética')):
                axis.imshow(img,cmap='gray',vmin=interval['p1'],vmax=interval['p99']); axis.contour(c.lung_mask,[.5],colors='#12a57a'); axis.set_title(title); axis.axis('off')
            ax[2].imshow(c.lung_mask,cmap='gray'); ax[2].set_title('ROI erodida'); ax[2].axis('off')
            fig.suptitle(f"{c.paths.case_id} | {status} | Δμ={delta:.3f} | Sat31={sat:.1f}% | LL={row['wavelet_ll_synthetic']:.3f}",fontsize=9)
            fig.tight_layout(); fig.savefig(previews/f'{n:02d}.png',dpi=150); plt.close(fig)
    with (a.output_dir/'mvp_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (a.output_dir/'mvp_protocol.json').write_text(json.dumps({'valid_pairs':len(valid),'glcm_scale':scale.as_dict(),'wavelet_interval':interval,'third_metric':'reserved'},indent=2),encoding='utf-8')
if __name__=='__main__': main()
