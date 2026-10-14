"""
Verificador de anomalias de brilho e densidade em imagens CT sintéticas (Dgen e Dsr).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk, erosion

from .load_data import discover_cases, load_image, CasePaths
from .validate_cases import validate_case, ValidCase, RejectedCase


def main() -> None:
    """Ponto de entrada do verificador de brilho e densidade."""
    parser = argparse.ArgumentParser(description="Verificador de anomalias de brilho e densidade em imagens CT.")
    parser.add_argument("--dgen-dir", type=Path, required=True, help="Diretório das imagens Dgen e máscaras.")
    parser.add_argument("--dsr-dir", type=Path, required=True, help="Diretório das imagens Dsr.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Diretório de saída para logs, CSVs e figuras.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 3. Descobrir e validar os casos
    cases_dgen = discover_cases(args.dgen_dir)
    valid_cases: list[ValidCase] = []
    
    for case_paths in cases_dgen:
        v = validate_case(case_paths)
        if isinstance(v, ValidCase):
            valid_cases.append(v)
            
    if not valid_cases:
        print("Nenhum caso válido encontrado após a validação.")
        return

    pooled_orig_pixels = []
    case_results = []
    
    # 4. Computar métricas básicas
    for v in valid_cases:
        orig_img = v.original
        dgen_img = v.synthetic
        mask = v.lung_mask  # A máscara já vem erodida do validate_case()
        
        # Carregar a imagem correspondente de Dsr
        dsr_path = args.dsr_dir / v.paths.synthetic.name
        if not dsr_path.exists():
            print(f"Aviso: Imagem Dsr não encontrada para o caso {v.paths.case_id} em {dsr_path}. Ignorando.")
            continue
            
        dsr_img = load_image(dsr_path)
        
        orig_pixels = orig_img[mask]
        dgen_pixels = dgen_img[mask]
        dsr_pixels = dsr_img[mask]
        
        pooled_orig_pixels.append(orig_pixels)
        
        orig_mean = np.mean(orig_pixels)
        
        # 5. Desvios pareados
        diff_dgen = np.mean(dgen_pixels) - orig_mean
        diff_dsr = np.mean(dsr_pixels) - orig_mean
        
        rel_diff_dgen = abs(diff_dgen) / abs(orig_mean) if orig_mean != 0 else 0.0
        rel_diff_dsr = abs(diff_dsr) / abs(orig_mean) if orig_mean != 0 else 0.0

        case_data = {
            "case_id": v.paths.case_id,
            "orig_mean": orig_mean,
            "orig_median": np.median(orig_pixels),
            "orig_std": np.std(orig_pixels),
            
            "dgen_mean": np.mean(dgen_pixels),
            "dgen_median": np.median(dgen_pixels),
            "dgen_std": np.std(dgen_pixels),
            
            "dsr_mean": np.mean(dsr_pixels),
            "dsr_median": np.median(dsr_pixels),
            "dsr_std": np.std(dsr_pixels),
            
            "diff_dgen": diff_dgen,
            "diff_dsr": diff_dsr,
            "rel_diff_dgen": rel_diff_dgen,
            "rel_diff_dsr": rel_diff_dsr,
            
            "orig_pixels": orig_pixels,
            "dgen_pixels": dgen_pixels,
            "dsr_pixels": dsr_pixels,
        }
        
        case_results.append(case_data)

    # 6. Calibração com Tukey k=3.0 nos pixels reais agrupados
    pooled_array = np.concatenate(pooled_orig_pixels)
    Q1_pixels = float(np.percentile(pooled_array, 25))
    Q3_pixels = float(np.percentile(pooled_array, 75))
    IQR_pixels = Q3_pixels - Q1_pixels
    clip_low = Q1_pixels - 3.0 * IQR_pixels
    clip_high = Q3_pixels + 3.0 * IQR_pixels
    
    # Calcular IQR das médias originais para as regras de decisão
    orig_means = [c["orig_mean"] for c in case_results]
    Q1_means = float(np.percentile(orig_means, 25))
    Q3_means = float(np.percentile(orig_means, 75))
    IQR_means = Q3_means - Q1_means
    
    hard_reject_diff = 2.0 * IQR_means
    manual_review_diff = 1.0 * IQR_means
    hard_reject_sat = 15.0  # %
    
    # 7. Aplicar regras de gatekeeper
    rows = []
    anomalous_rows = []
    for c in case_results:
        # Calcular saturation_ceiling (% de pixels >= clip_high)
        c["sat_orig"] = np.mean(c["orig_pixels"] >= clip_high) * 100.0
        c["sat_dgen"] = np.mean(c["dgen_pixels"] >= clip_high) * 100.0
        c["sat_dsr"] = np.mean(c["dsr_pixels"] >= clip_high) * 100.0
        
        # Limpar os arrays de pixels para economizar memória e não atrapalhar no CSV
        del c["orig_pixels"]
        del c["dgen_pixels"]
        del c["dsr_pixels"]
        
        # Decisão para Dgen
        if abs(c["diff_dgen"]) > hard_reject_diff or c["sat_dgen"] > hard_reject_sat:
            c["gate_dgen"] = "hard_reject"
        elif abs(c["diff_dgen"]) > manual_review_diff:
            c["gate_dgen"] = "manual_review"
        else:
            c["gate_dgen"] = "approved"
            
        # Decisão para Dsr
        if abs(c["diff_dsr"]) > hard_reject_diff or c["sat_dsr"] > hard_reject_sat:
            c["gate_dsr"] = "hard_reject"
        elif abs(c["diff_dsr"]) > manual_review_diff:
            c["gate_dsr"] = "manual_review"
        else:
            c["gate_dsr"] = "approved"
            
        rows.append(c)
        if c["gate_dgen"] in ("hard_reject", "manual_review") or c["gate_dsr"] in ("hard_reject", "manual_review"):
            anomalous_rows.append(c)
            
    # 8. Exportar resultados (CSV e JSON)
    fieldnames = [
        "case_id", 
        "orig_mean", "orig_median", "orig_std", "sat_orig",
        "dgen_mean", "dgen_median", "dgen_std", "diff_dgen", "rel_diff_dgen", "sat_dgen", "gate_dgen",
        "dsr_mean", "dsr_median", "dsr_std", "diff_dsr", "rel_diff_dsr", "sat_dsr", "gate_dsr"
    ]
    
    with open(args.output_dir / "brightness_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    with open(args.output_dir / "anomalous_cases.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(anomalous_rows)
        
    calibration_info = {
        "Q1_pixels": Q1_pixels,
        "Q3_pixels": Q3_pixels,
        "IQR_pixels": IQR_pixels,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "Q1_means": Q1_means,
        "Q3_means": Q3_means,
        "IQR_means": IQR_means,
        "threshold_hard_reject_diff": hard_reject_diff,
        "threshold_manual_review_diff": manual_review_diff,
        "threshold_hard_reject_sat_percent": hard_reject_sat
    }
    
    with open(args.output_dir / "calibration_info.json", "w", encoding="utf-8") as f:
        json.dump(calibration_info, f, indent=4)
        
    # 9. Gerar visualizações
    
    # Gráfico de Dispersão (Scatter Plot)
    plt.figure(figsize=(8, 6))
    orig_m = [c["orig_mean"] for c in rows]
    dgen_m = [c["dgen_mean"] for c in rows]
    dsr_m = [c["dsr_mean"] for c in rows]
    
    # Cores de acordo com as especificações (#4C78A8, #E45756, #54A24B)
    dgen_colors = ["#E45756" if c["gate_dgen"] == "hard_reject" else "#4C78A8" for c in rows]
    dsr_colors = ["#E45756" if c["gate_dsr"] == "hard_reject" else "#54A24B" for c in rows]
    
    plt.scatter(orig_m, dgen_m, c=dgen_colors, label="Dgen", alpha=0.7)
    plt.scatter(orig_m, dsr_m, c=dsr_colors, label="Dsr", alpha=0.7, marker="^")
    
    min_val = min(min(orig_m), min(dgen_m), min(dsr_m))
    max_val = max(max(orig_m), max(dgen_m), max(dsr_m))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", label="y = x (Referência)")
    
    plt.xlabel("Média de Intensidade (Original)")
    plt.ylabel("Média de Intensidade (Sintética)")
    plt.title("Comparação de Média de Intensidade: Original vs Sintéticas")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(fig_dir / "scatter_means.png", bbox_inches="tight", dpi=150)
    plt.close()
    
    # Gráfico de Barras (Bar Chart)
    plt.figure(figsize=(12, 6))
    rows_sorted = sorted(rows, key=lambda x: abs(x["diff_dgen"]), reverse=True)
    ids_sorted = [c["case_id"] for c in rows_sorted]
    diffs_sorted = [c["diff_dgen"] for c in rows_sorted]
    bar_colors = ["#E45756" if c["gate_dgen"] == "hard_reject" else "#4C78A8" for c in rows_sorted]
    
    plt.bar(ids_sorted, diffs_sorted, color=bar_colors)
    plt.axhline(hard_reject_diff, color="#E45756", linestyle="--", label="Hard Reject (+)")
    plt.axhline(-hard_reject_diff, color="#E45756", linestyle="--", label="Hard Reject (-)")
    plt.axhline(manual_review_diff, color="#F58518", linestyle=":", label="Manual Review (+)")
    plt.axhline(-manual_review_diff, color="#F58518", linestyle=":", label="Manual Review (-)")
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("Δμ (Dgen - Original)")
    plt.title("Desvio de Intensidade por Caso (Dgen vs Original)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "bar_diff_dgen.png", bbox_inches="tight", dpi=150)
    plt.close()
    
    # Histograma
    plt.figure(figsize=(8, 6))
    sat_orig = [c["sat_orig"] for c in rows]
    sat_dgen = [c["sat_dgen"] for c in rows]
    sat_dsr = [c["sat_dsr"] for c in rows]
    
    plt.hist(sat_orig, bins=20, alpha=0.5, label="Original", color="#4C78A8")
    plt.hist(sat_dgen, bins=20, alpha=0.5, label="Dgen", color="#E45756")
    plt.hist(sat_dsr, bins=20, alpha=0.5, label="Dsr", color="#54A24B")
    
    plt.xlabel("Saturação de Pixels (%)")
    plt.ylabel("Frequência (Nº de casos)")
    plt.title("Distribuição da Saturação de Pixels (≥ clip_high)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(fig_dir / "hist_saturation.png", bbox_inches="tight", dpi=150)
    plt.close()
    
    # 10. Imprimir resumo no stdout
    print("=== Resumo de Anomalias de Brilho/Densidade ===")
    if not anomalous_rows:
        print("Nenhum caso anômalo detectado. Todos os casos aprovados.")
    else:
        for c in anomalous_rows:
            print(f"Caso: {c['case_id']}")
            print(f"  Dgen: {c['gate_dgen'].upper():<14} | Δμ = {c['diff_dgen']:>7.4f} | Saturação = {c['sat_dgen']:>5.1f}%")
            print(f"  Dsr:  {c['gate_dsr'].upper():<14} | Δμ = {c['diff_dsr']:>7.4f} | Saturação = {c['sat_dsr']:>5.1f}%")


if __name__ == "__main__":
    main()
