"""Gera gráficos ROC para GLCM: Contraste, Energia e Combinação (LDA)."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_roc(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Calcula FPR, TPR e AUC usando a regra trapezoidal."""
    desc_indices = np.argsort(scores)[::-1]
    y_sorted = y_true[desc_indices]
    scores_sorted = scores[desc_indices]

    distinct_indices = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_sorted == 0)[threshold_idxs]

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    auc = float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))
    return fpr, tpr, auc


def plot_single_roc(
    ax: plt.Axes,
    df: pd.DataFrame,
    get_scores: callable,
    title: str,
    feature_desc: str,
) -> float:
    # 0 = Real, 1 = Sintético (Dsr)
    y_true_total = np.concatenate([np.zeros(len(df)), np.ones(len(df))])
    scores_total = get_scores(df)

    fpr_tot, tpr_tot, auc_tot = compute_roc(y_true_total, scores_total)

    # Linha da sorte / aleatório
    ax.plot([0, 1], [0, 1], color="#9e9e9e", linestyle=":", linewidth=1.2, label="Chance aleatória (AUC 0.500)")

    # Curva total
    ax.plot(fpr_tot, tpr_tot, color="black", linewidth=2.4, label=f"Total: AUC {auc_tot:.3f}")

    # Pacientes individuais
    patient_styles = {
        "P4P117": ("--", "Paciente A (25 cortes)"),
        "P4P105": ("-.", "Paciente B (11 cortes)"),
    }
    for pid, (ls, label) in patient_styles.items():
        sub = df[df["patient_id"] == pid].copy()
        y_p = np.concatenate([np.zeros(len(sub)), np.ones(len(sub))])
        sc_p = get_scores(sub)
        fpr_p, tpr_p, auc_p = compute_roc(y_p, sc_p)
        ax.plot(fpr_p, tpr_p, color="#424242", linestyle=ls, linewidth=1.4, label=f"{label}: AUC {auc_p:.3f}")

    # Ponto operacional com Especificidade = 1.00 (FPR = 0.0)
    idx_esp1 = np.where(fpr_tot == 0)[0][-1]
    sens_esp1 = tpr_tot[idx_esp1]
    ax.scatter([0.0], [sens_esp1], color="#d32f2f", marker="s", s=55, zorder=5,
               label=f"Limiar esp. 1.00 (sens. {sens_esp1:.2f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("1 − Especificidade (Taxa de Falsos Positivos)", fontsize=10)
    ax.set_ylabel("Sensibilidade (Taxa de Verdadeiros Positivos)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    return auc_tot


def main() -> None:
    csv_path = Path("resultados/comparacao_Dgen_Dsr/metricas_por_caso.csv")
    out_dir = Path("slides/imagens")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["patient_id"] = df["case_id"].str.extract(r"-(P4P\d+)")

    # 1. Função de score para Contraste (Dsr tem menor contraste -> score = -contraste)
    def score_contrast(data: pd.DataFrame) -> np.ndarray:
        return np.concatenate([-data["original_contrast"], -data["dsr_contrast"]])

    # 2. Função de score para Energia (Dsr tem textura mais homogênea/lisa -> score = +energia)
    def score_energy(data: pd.DataFrame) -> np.ndarray:
        return np.concatenate([data["original_energy"], data["dsr_energy"]])

    # 3. Função de score Combinado (Fisher Linear Discriminant Analysis analítico)
    # Ajustamos os pesos globais no dataset completo
    X_orig = np.column_stack([df["original_contrast"], df["original_energy"]])
    X_dsr = np.column_stack([df["dsr_contrast"], df["dsr_energy"]])
    X_all = np.vstack([X_orig, X_dsr])
    y_all = np.concatenate([np.zeros(len(df)), np.ones(len(df))])

    mu_all, std_all = X_all.mean(axis=0), X_all.std(axis=0)
    X_norm = (X_all - mu_all) / std_all

    m0 = X_norm[y_all == 0].mean(axis=0)
    m1 = X_norm[y_all == 1].mean(axis=0)
    Sw = np.cov(X_norm[y_all == 0], rowvar=False) + np.cov(X_norm[y_all == 1], rowvar=False)
    weights = np.linalg.solve(Sw, (m1 - m0))

    def score_combined(data: pd.DataFrame) -> np.ndarray:
        Xo = np.column_stack([data["original_contrast"], data["original_energy"]])
        Xd = np.column_stack([data["dsr_contrast"], data["dsr_energy"]])
        X = np.vstack([Xo, Xd])
        Xn = (X - mu_all) / std_all
        return Xn @ weights

    # Gerando os 3 gráficos individuais
    configs = [
        ("roc_glcm_contraste.png", score_contrast, "GLCM: Apenas Contraste", "Contraste"),
        ("roc_glcm_energia.png", score_energy, "GLCM: Apenas Energia", "Energia"),
        ("roc_glcm_combinado.png", score_combined, "GLCM: Contraste + Energia Combinados", "Combinado"),
    ]

    for fname, score_fn, title, desc in configs:
        fig, ax = plt.subplots(figsize=(6, 5.2), dpi=300)
        plot_single_roc(ax, df, score_fn, title, desc)
        fig.tight_layout()
        save_path = out_dir / fname
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Salvo: {save_path}")

    # Gerando o Painel com os 3 gráficos lado a lado (estilo artigo/apresentação)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    for ax, (_, score_fn, title, desc) in zip(axes, configs):
        plot_single_roc(ax, df, score_fn, title, desc)
    fig.suptitle("Capacidade Discriminativa de Imagens Sintéticas (Real vs. Dsr) via GLCM", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    panel_path = out_dir / "painel_roc_glcm_3_metodos.png"
    fig.savefig(panel_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Painel salvo: {panel_path}")


if __name__ == "__main__":
    main()
