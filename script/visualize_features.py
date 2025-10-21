#!/usr/bin/env python3
# scripts/visualize_features.py
"""
특성값 시각화 스크립트
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_extraction import FeatureExtractor
from src.io_utils import get_df_from_bin


def plot_time_series_and_fft(df: pd.DataFrame, 
                             features: dict,
                             save_path: Path = None):
    """
    시계열 + FFT 스펙트럼 시각화
    
    Args:
        df: 센서 데이터
        features: 추출된 특성값
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 1. 시계열 데이터
    ax1 = axes[0]
    ax1.plot(df['time'], df['value'], linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Acceleration [m/s²]')
    ax1.set_title(f'Time Series (PTP={features["ptp"]:.4f}, Noise={features["noise_level"]:.6f})')
    ax1.grid(True, alpha=0.3)
    
    # 평균선
    ax1.axhline(y=features['mean'], color='r', linestyle='--', 
                linewidth=1, label=f'Mean={features["mean"]:.6f}')
    ax1.legend()
    
    # 2. FFT 스펙트럼
    ax2 = axes[1]
    
    extractor = FeatureExtractor(sampling_rate=100)
    freqs, power = extractor.get_fft_spectrum(df['value'].values)
    
    # 주파수 범위 제한
    mask = (freqs >= 0.5) & (freqs <= 20)
    freqs_filtered = freqs[mask]
    power_filtered = power[mask]
    
    ax2.plot(freqs_filtered, power_filtered, linewidth=1)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Power')
    ax2.set_title('FFT Spectrum')
    ax2.grid(True, alpha=0.3)
    
    # 고유진동수 표시
    natural_freqs = features['natural_freqs']
    if natural_freqs:
        for i, freq in enumerate(natural_freqs, 1):
            idx = np.argmin(np.abs(freqs_filtered - freq))
            ax2.axvline(x=freq, color='r', linestyle='--', 
                       linewidth=1, alpha=0.7)
            ax2.text(freq, power_filtered[idx], f'  {i}차: {freq:.2f}Hz',
                    rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_features_distribution(features_df: pd.DataFrame,
                               save_path: Path = None):
    """
    특성값 분포 시각화
    
    Args:
        features_df: 특성값 DataFrame
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    columns = ['ptp', 'noise_level', 'rms', 'freq_1', 'freq_2', 'freq_3']
    titles = ['Peak-to-Peak', 'Noise Level', 'RMS', 
              '1st Natural Freq', '2nd Natural Freq', '3rd Natural Freq']
    
    for i, (col, title) in enumerate(zip(columns, titles)):
        if col in features_df.columns:
            data = features_df[col].dropna()
            
            axes[i].hist(data, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(title)
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'{title}\n(mean={data.mean():.4f}, std={data.std():.4f})')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_features.py <bin_file_or_csv>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if input_path.suffix == '.bin':
        # bin 파일 시각화
        df = get_df_from_bin(input_path)
        
        extractor = FeatureExtractor()
        features = extractor.extract_all(df)
        
        output_path = Path("output") / f"{input_path.stem}_plot.png"
        output_path.parent.mkdir(exist_ok=True)
        
        plot_time_series_and_fft(df, features, save_path=output_path)
    
    elif input_path.suffix in ['.csv', '.pkl']:
        # 특성값 분포 시각화
        if input_path.suffix == '.csv':
            features_df = pd.read_csv(input_path)
        else:
            features_df = pd.read_pickle(input_path)
        
        output_path = Path("output") / f"{input_path.stem}_distribution.png"
        output_path.parent.mkdir(exist_ok=True)
        
        plot_features_distribution(features_df, save_path=output_path)
    
    else:
        print(f"Unsupported file type: {input_path.suffix}")