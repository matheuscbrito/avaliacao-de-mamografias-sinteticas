from __future__ import annotations

import numpy as np

class GLCMExtractor:
    """
    Extrator de atributos de textura usando a Matriz de Co-ocorrência de Níveis de Cinza (GLCM).
    
    Esta classe é autocontida e não depende de outros módulos do projeto.
    Ela calcula os atributos usando matrizes simétricas para 4 direções e inclui
    quantização própria usando calibração de escala robusta com cercas de Tukey.
    """
    
    def __init__(
        self, 
        levels: int = 32, 
        clip_low: float | None = None, 
        clip_high: float | None = None, 
        tukey_k: float = 3.0
    ):
        """
        Inicializa o extrator GLCM.
        
        Args:
            levels: Número de níveis de cinza após a quantização. Padrão é 32.
            clip_low: Limite inferior para recorte de intensidades.
            clip_high: Limite superior para recorte de intensidades.
            tukey_k: Fator k de Tukey usado na calibração (para documentação e uso interno). Padrão é 3.0.
        """
        self.levels = levels
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.tukey_k = tukey_k
        
        # Offsets para as 4 direções: 0, 45, 90 e 135 graus.
        # Usa a mesma lógica baseada nos offsets (dy, dx).
        self._offsets = ((0, 1), (-1, 1), (-1, 0), (-1, -1))
        
    def calibrate_from_pixels(self, pixels: np.ndarray) -> None:
        """
        Calibra os limites de recorte (clip_low e clip_high) usando o método
        das cercas de Tukey aplicado a um vetor 1D de pixels reais do pulmão.
        
        Args:
            pixels: Array 1D com os valores de intensidade dos pixels.
        """
        if pixels.size == 0:
            raise ValueError("O array de pixels não pode estar vazio para calibração.")
            
        q1 = float(np.percentile(pixels, 25))
        q3 = float(np.percentile(pixels, 75))
        iqr = q3 - q1
        
        self.clip_low = q1 - self.tukey_k * iqr
        self.clip_high = q3 + self.tukey_k * iqr

    def extract(self, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        """
        Extrai os atributos GLCM e de saturação de uma imagem, considerando apenas 
        a região da máscara.
        
        Args:
            image: Imagem de entrada.
            mask: Máscara binária ou booleana indicando a Região de Interesse (ROI).
            
        Returns:
            Dicionário com os atributos calculados (todas as chaves com prefixo 'glcm_').
        """
        if self.clip_low is None or self.clip_high is None:
            raise ValueError("Limites de recorte não definidos. Defina-os no construtor ou use 'calibrate_from_pixels'.")
            
        # Garante que a máscara seja do tipo booleano
        mask_bool = mask.astype(bool)
        
        if not np.any(mask_bool):
            return self._empty_results()
            
        # 1. Recorte (Clipping)
        clipped = np.clip(image, self.clip_low, self.clip_high)
        
        # 2. Quantização
        # Fórmula: floor((I - clip_low) / (clip_high - clip_low) * (levels - 1))
        range_val = self.clip_high - self.clip_low
        if range_val <= 0:
            range_val = 1e-8  # Evita divisão por zero se a imagem for constante
            
        quantized = np.floor((clipped - self.clip_low) / range_val * (self.levels - 1)).astype(np.int32)
        # Garante que os valores fiquem estritamente dentro de [0, levels-1]
        quantized = np.clip(quantized, 0, self.levels - 1)
        
        # 3. Métricas de saturação (calculadas diretamente na imagem quantizada dentro da máscara)
        roi_pixels = quantized[mask_bool]
        total_roi_pixels = roi_pixels.size
        
        saturation_floor = float(np.sum(roi_pixels == 0) / total_roi_pixels)
        saturation_ceiling = float(np.sum(roi_pixels == (self.levels - 1)) / total_roi_pixels)
        
        # 4. Cálculo da matriz GLCM e dos atributos
        features = {
            'glcm_contrast': 0.0,
            'glcm_energy': 0.0,
            'glcm_dissimilarity': 0.0,
            'glcm_homogeneity': 0.0,
            'glcm_entropy': 0.0,
            'glcm_correlation': 0.0,
            'glcm_gray_mean': 0.0,
            'glcm_gray_variance': 0.0,
        }
        
        valid_angles = 0
        levels_range = np.arange(self.levels)
        i_grid, j_grid = np.meshgrid(levels_range, levels_range, indexing='ij')
        
        # Diferenças pré-calculadas para desempenho
        diff = i_grid - j_grid
        diff_sq = diff ** 2
        diff_abs = np.abs(diff)
        
        for dy, dx in self._offsets:
            glcm = self._compute_glcm(quantized, mask_bool, dy, dx)
            
            sum_glcm = np.sum(glcm)
            if sum_glcm == 0:
                continue
                
            # Normaliza a GLCM para obter P(i,j)
            p = glcm / sum_glcm
            
            # --- Contrast, Energy, Dissimilarity, Homogeneity ---
            features['glcm_contrast'] += float(np.sum(p * diff_sq))
            features['glcm_energy'] += float(np.sum(p ** 2))
            features['glcm_dissimilarity'] += float(np.sum(p * diff_abs))
            features['glcm_homogeneity'] += float(np.sum(p / (1.0 + diff_sq)))
            
            # --- Entropy ---
            # Usa apenas p > 0 para evitar log2(0)
            mask_p = p > 0
            features['glcm_entropy'] += float(-np.sum(p[mask_p] * np.log2(p[mask_p])))
            
            # --- Gray Mean & Variance (Marginais) ---
            # Considerando matriz simétrica, a marginal de 'i' serve.
            p_x = np.sum(p, axis=1)
            
            mu_i = float(np.sum(levels_range * p_x))
            features['glcm_gray_mean'] += mu_i
            
            var_i = float(np.sum(((levels_range - mu_i) ** 2) * p_x))
            features['glcm_gray_variance'] += var_i
            
            # --- Correlation ---
            if var_i > 1e-8:
                corr = float(np.sum((i_grid - mu_i) * (j_grid - mu_i) * p) / var_i)
                features['glcm_correlation'] += corr
            else:
                # Variância próxima a zero implica imagem quase constante, correlação máxima.
                features['glcm_correlation'] += 1.0
                
            valid_angles += 1
            
        # Calcula a média das métricas entre as direções avaliadas
        if valid_angles > 0:
            for k in features:
                features[k] /= valid_angles
                
        # Adiciona as métricas que não dependem da média por ângulo
        features['glcm_saturation_floor'] = saturation_floor
        features['glcm_saturation_ceiling'] = saturation_ceiling
        
        return features
        
    def _compute_glcm(self, image: np.ndarray, mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """
        Calcula a matriz de co-ocorrência simétrica para um offset (dy, dx).
        
        Args:
            image: Imagem quantizada do tipo int32.
            mask: Máscara booleana determinando as regiões válidas.
            dy: Offset de linha (eixo y).
            dx: Offset de coluna (eixo x).
            
        Returns:
            Matriz GLCM de tamanho levels x levels contendo o número de pares encontrados.
        """
        rows, cols = image.shape
        glcm = np.zeros((self.levels, self.levels), dtype=np.float64)
        
        y_start = max(0, -dy)
        y_end = min(rows, rows - dy)
        x_start = max(0, -dx)
        x_end = min(cols, cols - dx)
        
        if y_start >= y_end or x_start >= x_end:
            return glcm
            
        # Extrai as áreas base e deslocada
        img_ref = image[y_start:y_end, x_start:x_end]
        img_nbr = image[y_start+dy:y_end+dy, x_start+dx:x_end+dx]
        
        mask_ref = mask[y_start:y_end, x_start:x_end]
        mask_nbr = mask[y_start+dy:y_end+dy, x_start+dx:x_end+dx]
        
        # Mantém apenas os pares que se encontram inteiramente na ROI
        valid_pairs = mask_ref & mask_nbr
        
        if not np.any(valid_pairs):
            return glcm
            
        vals_ref = img_ref[valid_pairs]
        vals_nbr = img_nbr[valid_pairs]
        
        # Para ser simétrica, soma-se tanto (ref, nbr) quanto (nbr, ref)
        np.add.at(glcm, (vals_ref, vals_nbr), 1)
        np.add.at(glcm, (vals_nbr, vals_ref), 1)
        
        return glcm
        
    def _empty_results(self) -> dict[str, float]:
        """Retorna os atributos com valor zero ou apropriado quando a ROI for vazia."""
        return {
            'glcm_contrast': 0.0,
            'glcm_energy': 0.0,
            'glcm_dissimilarity': 0.0,
            'glcm_homogeneity': 0.0,
            'glcm_entropy': 0.0,
            'glcm_correlation': 0.0,
            'glcm_gray_mean': 0.0,
            'glcm_gray_variance': 0.0,
            'glcm_saturation_floor': 0.0,
            'glcm_saturation_ceiling': 0.0,
        }
