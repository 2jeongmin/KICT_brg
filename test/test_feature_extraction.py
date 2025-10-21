# tests/test_feature_extraction.py
"""
feature_extraction.py 테스트
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_extraction import (
    FeatureExtractor, 
    BatchFeatureExtractor,
    extract_features_quick,
    extract_features_batch
)
from src.io_utils import get_df_from_bin
from src.zip_handler import ZipHandler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_file():
    """단일 파일 특성값 추출 테스트"""
    print("\n=== Test 1: Single File Feature Extraction ===")
    
    # 테스트 데이터 생성 (실제 파일이 없는 경우)
    n_samples = 360000
    t = np.linspace(0, 3600, n_samples)
    
    # 시뮬레이션: 2Hz 지배 주파수 + 노이즈
    signal = 0.01 * np.sin(2 * np.pi * 2.0 * t) + 0.001 * np.random.randn(n_samples)
    
    times = pd.date_range('2024-01-01 00:00:00', periods=n_samples, freq='10ms')
    df = DataFrame({
        'time': times,
        'value': signal
    })
    
    print(f"Test data: {len(df)} records")
    
    # 특성값 추출
    extractor = FeatureExtractor(sampling_rate=100)
    features = extractor.extract_all(df)
    
    print("\nExtracted features:")
    print(f"  PTP: {features['ptp']:.6f}")
    print(f"  Mean: {features['mean']:.6f}")
    print(f"  Std: {features['std']:.6f}")
    print(f"  RMS: {features['rms']:.6f}")
    print(f"  Noise: {features['noise_level']:.6f}")
    print(f"  Natural freqs: {features['natural_freqs']}")
    print(f"  Valid: {features['is_valid']}")


def test_real_file():
    """실제 bin 파일 테스트"""
    print("\n=== Test 2: Real File Feature Extraction ===")
    
    # 실제 파일 경로 (존재하는 경우만 테스트)
    test_file = Path("data/smartcs_1_DNA21006_2024032620.bin")
    
    if not test_file.exists():
        print(f"Skip: File not found - {test_file}")
        return
    
    # 데이터 로드
    df = get_df_from_bin(test_file)
    print(f"Loaded: {len(df)} records")
    
    # 특성값 추출
    features = extract_features_quick(df)
    
    print("\nFeatures:")
    for key, value in features.items():
        if key not in ['validation_issues']:
            print(f"  {key}: {value}")


def test_batch_extraction():
    """배치 특성값 추출 테스트"""
    print("\n=== Test 3: Batch Feature Extraction ===")
    
    # zip 파일에서 데이터 로드
    root_dir = Path("Z:/05. Data/01. Under_Process/027. KICT_BMAPS/upload")
    
    if not root_dir.exists():
        print(f"Skip: Directory not found - {root_dir}")
        return
    
    # zip 처리
    handler = ZipHandler(mode='stream')
    
    all_features = []
    for zip_results in handler.process_all_zips(root_dir, max_files=3):
        # 특성값 추출
        features_df = extract_features_batch(zip_results)
        all_features.append(features_df)
        
        print(f"Processed batch: {len(features_df)} files")
        print(features_df[['sensor_id', 'hour', 'ptp', 'freq_1', 'freq_2', 'freq_3']].head())
    
    # 전체 결과 결합
    if all_features:
        combined_df = pd.concat(all_features, ignore_index=True)
        print(f"\nTotal processed: {len(combined_df)} files")
        
        # 통계 요약
        print("\nStatistics Summary:")
        print(combined_df[['ptp', 'noise_level', 'freq_1']].describe())


def test_fft_analysis():
    """FFT 분석 테스트"""
    print("\n=== Test 4: FFT Analysis ===")
    
    # 테스트 신호 생성: 2Hz + 5Hz + 10Hz
    n_samples = 360000
    sampling_rate = 100
    t = np.linspace(0, n_samples/sampling_rate, n_samples)
    
    signal = (
        0.01 * np.sin(2 * np.pi * 2.0 * t) +   # 2Hz (강한 신호)
        0.005 * np.sin(2 * np.pi * 5.0 * t) +  # 5Hz (중간)
        0.003 * np.sin(2 * np.pi * 10.0 * t) + # 10Hz (약한 신호)
        0.001 * np.random.randn(n_samples)      # 노이즈
    )
    
    times = pd.date_range('2024-01-01 00:00:00', periods=n_samples, freq='10ms')
    df = DataFrame({
        'time': times,
        'value': signal
    })
    
    # FFT 분석
    extractor = FeatureExtractor(
        sampling_rate=sampling_rate,
        fft_freq_range=(0.5, 20),
        fft_n_peaks=3
    )
    
    # 고유진동수 추출
    freqs = extractor.extract_natural_frequencies(df['value'].values)
    
    print(f"Expected frequencies: 2.0, 5.0, 10.0 Hz")
    print(f"Detected frequencies: {[f'{f:.2f}' for f in freqs]} Hz")
    
    # 전체 스펙트럼
    xf, power = extractor.get_fft_spectrum(df['value'].values)
    
    # 피크 위치 출력
    peak_indices = np.argsort(power)[-5:][::-1]  # 상위 5개
    print("\nTop 5 FFT peaks:")
    for idx in peak_indices:
        print(f"  {xf[idx]:.2f} Hz: Power = {power[idx]:.6f}")


def test_validation():
    """데이터 검증 테스트"""
    print("\n=== Test 5: Data Validation ===")
    
    extractor = FeatureExtractor(validate=True)
    
    # 테스트 케이스 1: 정상 데이터
    print("\n[Case 1: Normal data]")
    df_normal = DataFrame({
        'time': pd.date_range('2024-01-01', periods=360000, freq='10ms'),
        'value': np.random.randn(360000) * 0.01
    })
    validation = extractor.validate_data(df_normal)
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Issues: {validation['issues']}")
    print(f"  Warnings: {validation['warnings']}")
    
    # 테스트 케이스 2: 데이터 부족
    print("\n[Case 2: Too few records]")
    df_few = DataFrame({
        'time': pd.date_range('2024-01-01', periods=100000, freq='10ms'),
        'value': np.random.randn(100000) * 0.01
    })
    validation = extractor.validate_data(df_few)
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Issues: {validation['issues']}")
    print(f"  Warnings: {validation['warnings']}")
    
    # 테스트 케이스 3: 범위 초과
    print("\n[Case 3: Out of range values]")
    df_range = DataFrame({
        'time': pd.date_range('2024-01-01', periods=360000, freq='10ms'),
        'value': np.random.randn(360000) * 150  # -100~100 범위 초과
    })
    validation = extractor.validate_data(df_range)
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Issues: {validation['issues']}")
    print(f"  Warnings: {validation['warnings']}")
    
    # 테스트 케이스 4: NaN 포함
    print("\n[Case 4: NaN values]")
    df_nan = DataFrame({
        'time': pd.date_range('2024-01-01', periods=360000, freq='10ms'),
        'value': np.random.randn(360000) * 0.01
    })
    df_nan.loc[1000:1100, 'value'] = np.nan
    validation = extractor.validate_data(df_nan)
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Issues: {validation['issues']}")
    print(f"  Warnings: {validation['warnings']}")


def test_noise_methods():
    """다양한 노이즈 레벨 계산 방법 비교"""
    print("\n=== Test 6: Noise Level Methods Comparison ===")
    
    # 테스트 신호: 정적 구간 + 동적 구간
    n_samples = 360000
    signal = np.zeros(n_samples)
    
    # 앞부분: 정적 (노이즈만)
    signal[:180000] = 0.001 * np.random.randn(180000)
    
    # 뒷부분: 동적 (신호 + 노이즈)
    t = np.linspace(0, 1800, 180000)
    signal[180000:] = 0.01 * np.sin(2 * np.pi * 2.0 * t) + 0.001 * np.random.randn(180000)
    
    df = DataFrame({
        'time': pd.date_range('2024-01-01', periods=n_samples, freq='10ms'),
        'value': signal
    })
    
    methods = ['static_std', 'full_std']
    
    for method in methods:
        extractor = FeatureExtractor(noise_method=method)
        noise = extractor.extract_noise_level(df['value'].values)
        print(f"  {method:15s}: {noise:.6f}")


def test_performance():
    """성능 테스트"""
    print("\n=== Test 7: Performance Test ===")
    
    import time
    
    # 테스트 데이터
    n_samples = 360000
    df = DataFrame({
        'time': pd.date_range('2024-01-01', periods=n_samples, freq='10ms'),
        'value': np.random.randn(n_samples) * 0.01
    })
    
    extractor = FeatureExtractor(sampling_rate=100)
    
    # 워밍업
    _ = extractor.extract_all(df)
    
    # 성능 측정
    n_iterations = 10
    start = time.time()
    
    for _ in range(n_iterations):
        features = extractor.extract_all(df)
    
    elapsed = time.time() - start
    avg_time = elapsed / n_iterations
    
    print(f"  Average time per file: {avg_time:.3f} seconds")
    print(f"  Processing speed: {n_samples/avg_time:,.0f} samples/sec")
    print(f"  Estimated time for 50,000 files: {(avg_time * 50000 / 3600):.1f} hours")


if __name__ == "__main__":
    print("="*60)
    print("Feature Extraction Tests")
    print("="*60)
    
    # 실행할 테스트 선택
    test_single_file()
    test_fft_analysis()
    test_validation()
    test_noise_methods()
    test_performance()
    
    # 실제 파일 테스트 (파일이 있는 경우만)
    # test_real_file()
    # test_batch_extraction()
    
    print("\n" + "="*60)
    print("All tests completed")
    print("="*60)