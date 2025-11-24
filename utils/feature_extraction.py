# src/feature_extraction.py
"""
센서 데이터 특성값 추출 모듈

추출 특성값:
- 가속도 센서:
  * Peak-to-Peak (PTP): 최대-최소 진폭
  * 통계값: mean, std, rms, kurtosis, skewness
  * 노이즈 레벨: 정적 구간의 표준편차
  * 고유진동수: FFT 기반 지배 주파수 검출 (최대 3개)

- 정적 센서:
  * mean, ptp, noise_level
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    센서 데이터 특성값 추출 클래스
    """
    
    def __init__(self, 
                 sampling_rate: int = 100,
                 fft_freq_range: Tuple[float, float] = (0.5, 20),
                 fft_n_peaks: int = 3,
                 fft_prominence_ratio: float = 0.1,
                 noise_method: str = 'static_std',
                 noise_static_threshold: float = 0.01,
                 validate: bool = True):
        """
        Args:
            sampling_rate: 샘플링 주파수 [Hz]
            fft_freq_range: FFT 분석 주파수 범위 [Hz] (min, max)
            fft_n_peaks: 추출할 고유진동수 개수
            fft_prominence_ratio: FFT 피크 검출 임계값 (최대값 대비 비율)
            noise_method: 노이즈 레벨 계산 방법 ('static_std', 'full_std', 'snr')
            noise_static_threshold: 정적 구간 판단 기준 [m/s²]
            validate: 데이터 검증 수행 여부
        """
        self.sampling_rate = sampling_rate
        self.fft_freq_range = fft_freq_range
        self.fft_n_peaks = fft_n_peaks
        self.fft_prominence_ratio = fft_prominence_ratio
        self.noise_method = noise_method
        self.noise_static_threshold = noise_static_threshold
        self.validate = validate
    
    
    def extract_all(self, 
                    df: DataFrame, 
                    sensor_type: str = 'acceleration') -> Dict:
        """
        모든 특성값 추출 (메인 함수)
        
        Args:
            df: 센서 데이터 DataFrame (columns=['time', 'value'])
            sensor_type: 센서 타입 ('acceleration' or 'static')
        
        Returns:
            특성값 딕셔너리
        """
        if df is None or len(df) == 0:
            logger.warning("Empty DataFrame provided")
            return self._get_empty_features(sensor_type)
        
        try:
            # 데이터 검증
            if self.validate:
                validation = self.validate_data(df)
                if not validation['is_valid']:
                    logger.warning(f"Data validation failed: {validation['issues']}")
            else:
                validation = {'is_valid': True, 'issues': []}
            
            # 센서 타입별 특성값 추출
            if sensor_type == 'acceleration':
                features = self._extract_acceleration_features(df)
            elif sensor_type == 'static':
                features = self._extract_static_features(df)
            else:
                raise ValueError(f"Unknown sensor type: {sensor_type}")
            
            # 검증 결과 추가
            features['is_valid'] = validation['is_valid']
            features['validation_issues'] = validation.get('issues', [])
            features['data_count'] = len(df)
            
            return features
        
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._get_empty_features(sensor_type, error=str(e))
    
    
    def _extract_acceleration_features(self, df: DataFrame) -> Dict:
        """
        가속도 센서 특성값 추출
        
        Args:
            df: 가속도 데이터 DataFrame
        
        Returns:
            특성값 딕셔너리
        """
        values = df['value'].values
        
        features = {}
        
        # 1. 기본 통계값
        features.update(self.extract_statistics(values))
        
        # 2. Peak-to-Peak
        features['ptp'] = self.extract_ptp(values)
        
        # 3. RMS
        features['rms'] = self.extract_rms(values)
        
        # 4. 노이즈 레벨
        features['noise_level'] = self.extract_noise_level(values)
        
        # 5. 고유진동수 (FFT)
        features['natural_freqs'] = self.extract_natural_frequencies(values)
        
        # 6. Kurtosis & Skewness
        features['kurtosis'] = self.extract_kurtosis(values)
        features['skewness'] = self.extract_skewness(values)
        
        return features
    
    
    def _extract_static_features(self, df: DataFrame) -> Dict:
        """
        정적 센서 특성값 추출 (온도, 습도, 변위, 균열 등)
        
        Args:
            df: 정적 센서 데이터 DataFrame
        
        Returns:
            특성값 딕셔너리
        """
        values = df['value'].values
        
        features = {
            'mean': float(np.mean(values)),
            'ptp': self.extract_ptp(values),
            'noise_level': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
        
        return features
    
    
    def extract_statistics(self, values: np.ndarray) -> Dict[str, float]:
        """
        기본 통계값 추출
        
        Args:
            values: 센서 값 배열
        
        Returns:
            통계값 딕셔너리 (mean, std, min, max, median)
        """
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    
    
    def extract_ptp(self, values: np.ndarray) -> float:
        """
        Peak-to-Peak (최대-최소) 진폭 계산
        
        Args:
            values: 센서 값 배열
        
        Returns:
            PTP 값
        """
        return float(np.ptp(values))
    
    
    def extract_rms(self, values: np.ndarray) -> float:
        """
        RMS (Root Mean Square) 계산
        
        Args:
            values: 센서 값 배열
        
        Returns:
            RMS 값
        """
        return float(np.sqrt(np.mean(values**2)))
    
    
    def extract_kurtosis(self, values: np.ndarray) -> float:
        """
        첨도(Kurtosis) 계산
        
        Args:
            values: 센서 값 배열
        
        Returns:
            Kurtosis 값
        """
        try:
            return float(kurtosis(values))
        except Exception as e:
            logger.warning(f"Kurtosis calculation failed: {e}")
            return np.nan
    
    
    def extract_skewness(self, values: np.ndarray) -> float:
        """
        왜도(Skewness) 계산
        
        Args:
            values: 센서 값 배열
        
        Returns:
            Skewness 값
        """
        try:
            return float(skew(values))
        except Exception as e:
            logger.warning(f"Skewness calculation failed: {e}")
            return np.nan
    
    
    def extract_noise_level(self, values: np.ndarray) -> float:
        """
        노이즈 레벨 계산
        
        방법:
        - 'static_std': 정적 구간(거의 변화 없는 구간)의 표준편차
        - 'full_std': 전체 데이터의 표준편차
        - 'snr': Signal-to-Noise Ratio 기반
        
        Args:
            values: 센서 값 배열
        
        Returns:
            노이즈 레벨
        """
        if self.noise_method == 'static_std':
            # 정적 구간 찾기: 변화량이 threshold 이하인 구간
            diffs = np.abs(np.diff(values))
            static_mask = diffs < self.noise_static_threshold
            
            if np.sum(static_mask) > 10:  # 최소 10개 이상의 정적 데이터
                static_values = values[:-1][static_mask]
                noise = float(np.std(static_values))
            else:
                # 정적 구간이 충분하지 않으면 전체 std 사용
                noise = float(np.std(values))
                logger.debug("Not enough static data, using full std for noise level")
            
        elif self.noise_method == 'full_std':
            noise = float(np.std(values))
        
        elif self.noise_method == 'snr':
            # SNR 기반: signal_power / noise_power
            signal_power = np.mean(values**2)
            noise_power = np.var(values)
            if noise_power > 0:
                snr = signal_power / noise_power
                noise = float(1.0 / snr) if snr > 0 else np.inf
            else:
                noise = 0.0
        
        else:
            raise ValueError(f"Unknown noise method: {self.noise_method}")
        
        return noise
    
    
    def extract_natural_frequencies(self, 
                                   values: np.ndarray,
                                   return_powers: bool = False) -> List[float]:
        """
        FFT 기반 고유진동수 추출
        
        Args:
            values: 가속도 값 배열
            return_powers: True면 (주파수, 파워) 튜플 반환
        
        Returns:
            고유진동수 리스트 [Hz] (최대 n_peaks개, 진폭 순 정렬)
        """
        try:
            n = len(values)
            
            # FFT 수행
            yf = rfft(values)
            xf = rfftfreq(n, 1/self.sampling_rate)
            
            # 파워 스펙트럼 (진폭)
            power = np.abs(yf)
            
            # 주파수 범위 필터링
            freq_min, freq_max = self.fft_freq_range
            mask = (xf >= freq_min) & (xf <= freq_max)
            
            if not np.any(mask):
                logger.warning(f"No frequencies in range {self.fft_freq_range}")
                return []
            
            xf_filtered = xf[mask]
            power_filtered = power[mask]
            
            # 피크 검출
            prominence_threshold = power_filtered.max() * self.fft_prominence_ratio
            
            peaks, properties = find_peaks(
                power_filtered,
                prominence=prominence_threshold,
                distance=int(0.1 / (xf_filtered[1] - xf_filtered[0]))  # 최소 0.1Hz 간격
            )
            
            if len(peaks) == 0:
                logger.debug("No peaks found in FFT")
                return []
            
            # 진폭 순으로 정렬하여 상위 n개 선택
            peak_powers = power_filtered[peaks]
            top_indices = np.argsort(peak_powers)[::-1][:self.fft_n_peaks]
            
            # 주파수 추출 (낮은 주파수 순으로 정렬)
            top_peaks = peaks[top_indices]
            natural_freqs = xf_filtered[top_peaks]
            natural_freqs = np.sort(natural_freqs)  # 주파수 순 정렬
            
            if return_powers:
                powers = power_filtered[top_peaks]
                return list(zip(natural_freqs.tolist(), powers.tolist()))
            else:
                return natural_freqs.tolist()
        
        except Exception as e:
            logger.error(f"FFT analysis failed: {e}")
            return []
    
    
    def validate_data(self, df: DataFrame) -> Dict:
        """
        데이터 품질 검증
        
        Args:
            df: 센서 데이터 DataFrame
        
        Returns:
            검증 결과 딕셔너리
            {
                'is_valid': bool,
                'issues': List[str],
                'warnings': List[str]
            }
        """
        issues = []
        warnings = []
        
        if len(df) == 0:
            issues.append("empty_data")
            return {'is_valid': False, 'issues': issues, 'warnings': warnings}
        
        values = df['value'].values
        
        # 1. 데이터 개수 체크 (시간당 360,000개 기준)
        if len(df) > 400000:
            warnings.append("too_many_records")
        elif len(df) < 320000:
            warnings.append("too_few_records")
        
        # 2. 시간 간격 체크 (100Hz → ~10ms)
        if len(df) >= 2:
            time_diff = (df['time'].iloc[1] - df['time'].iloc[0]).total_seconds() * 1000
            
            if time_diff == 0:
                issues.append("zero_time_delta")
            elif time_diff < 9 or time_diff > 11:
                warnings.append(f"abnormal_time_delta_{time_diff:.2f}ms")
        
        # 3. 값 유효성 체크
        if not np.all(np.isfinite(values)):
            issues.append("non_finite_values")
        
        # 4. 값 범위 체크 (가속도: -100 ~ 100 m/s²)
        min_val, max_val = values.min(), values.max()
        if min_val < -100 or max_val > 100:
            warnings.append(f"out_of_range_[{min_val:.2f},{max_val:.2f}]")
        
        # 5. 시간 정렬 체크
        if not df['time'].is_monotonic_increasing:
            issues.append("time_not_sorted")
        
        # 6. 중복 시간 체크
        if df['time'].duplicated().any():
            warnings.append("duplicate_timestamps")
        
        is_valid = len(issues) == 0
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'warnings': warnings
        }
    
    
    def _get_empty_features(self, sensor_type: str, error: Optional[str] = None) -> Dict:
        """
        빈 특성값 딕셔너리 반환 (에러 시)
        
        Args:
            sensor_type: 센서 타입
            error: 에러 메시지
        
        Returns:
            빈 특성값 딕셔너리 (모든 값 None)
        """
        if sensor_type == 'acceleration':
            features = {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'median': None,
                'ptp': None,
                'rms': None,
                'noise_level': None,
                'natural_freqs': [],
                'kurtosis': None,
                'skewness': None,
                'is_valid': False,
                'validation_issues': [error] if error else [],
                'data_count': 0
            }
        else:  # static
            features = {
                'mean': None,
                'ptp': None,
                'noise_level': None,
                'min': None,
                'max': None,
                'is_valid': False,
                'validation_issues': [error] if error else [],
                'data_count': 0
            }
        
        return features
    
    
    def get_fft_spectrum(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        전체 FFT 스펙트럼 반환 (시각화용)
        
        Args:
            values: 가속도 값 배열
        
        Returns:
            (주파수 배열, 파워 배열)
        """
        n = len(values)
        yf = rfft(values)
        xf = rfftfreq(n, 1/self.sampling_rate)
        power = np.abs(yf)
        
        return xf, power


class BatchFeatureExtractor:
    """
    배치 처리용 특성값 추출 클래스
    """
    
    def __init__(self, extractor: FeatureExtractor):
        """
        Args:
            extractor: FeatureExtractor 인스턴스
        """
        self.extractor = extractor
        self.results = []
    
    
    def extract_from_zip_results(self, 
                                 zip_results: List[Dict],
                                 sensor_type: str = 'acceleration') -> List[Dict]:
        """
        zip_handler 결과로부터 특성값 추출
        
        Args:
            zip_results: zip_handler.process_single_zip() 결과
            sensor_type: 센서 타입
        
        Returns:
            특성값 리스트
        """
        features_list = []
        
        for result in zip_results:
            try:
                df = result['df']
                
                # 특성값 추출
                features = self.extractor.extract_all(df, sensor_type=sensor_type)
                
                # 메타데이터 추가
                features.update({
                    'sensor_id': result['sensor_id'],
                    'hour': result['hour'],
                    'source_zip': result['zip_file'],
                    'source_bin': result['bin_file'],
                    'file_size': result['file_size']
                })
                
                features_list.append(features)
                
            except Exception as e:
                logger.error(f"Failed to extract features from {result.get('bin_file', 'unknown')}: {e}")
                continue
        
        return features_list
    
    
    def to_dataframe(self, features_list: List[Dict]) -> DataFrame:
        """
        특성값 리스트를 DataFrame으로 변환
        
        Args:
            features_list: 특성값 딕셔너리 리스트
        
        Returns:
            특성값 DataFrame
        """
        # natural_freqs는 배열이므로 별도 처리
        records = []
        
        for features in features_list:
            record = features.copy()
            
            # natural_freqs를 개별 컬럼으로 분리
            nat_freqs = record.pop('natural_freqs', [])
            for i in range(3):  # 최대 3개
                if i < len(nat_freqs):
                    record[f'freq_{i+1}'] = nat_freqs[i]
                else:
                    record[f'freq_{i+1}'] = None
            
            # validation_issues를 문자열로 변환
            if 'validation_issues' in record:
                record['validation_issues'] = ','.join(record['validation_issues'])
            
            records.append(record)
        
        df = DataFrame(records)
        return df


# 편의 함수들

def extract_features_quick(df: DataFrame, 
                           sampling_rate: int = 100,
                           sensor_type: str = 'acceleration') -> Dict:
    """
    빠른 특성값 추출 (기본 설정)
    
    Args:
        df: 센서 데이터 DataFrame
        sampling_rate: 샘플링 주파수
        sensor_type: 센서 타입
    
    Returns:
        특성값 딕셔너리
    
    Example:
        >>> df = get_df_from_bin("data.bin")
        >>> features = extract_features_quick(df)
        >>> print(f"PTP: {features['ptp']:.4f}")
        >>> print(f"Natural frequencies: {features['natural_freqs']}")
    """
    extractor = FeatureExtractor(sampling_rate=sampling_rate)
    return extractor.extract_all(df, sensor_type=sensor_type)


def extract_features_batch(results_list: List[Dict],
                           sampling_rate: int = 100,
                           sensor_type: str = 'acceleration') -> DataFrame:
    """
    배치 특성값 추출 및 DataFrame 변환
    
    Args:
        results_list: zip_handler 결과 리스트
        sampling_rate: 샘플링 주파수
        sensor_type: 센서 타입
    
    Returns:
        특성값 DataFrame
    
    Example:
        >>> from src.zip_handler import ZipHandler
        >>> handler = ZipHandler(mode='stream')
        >>> results = handler.process_single_zip(zip_file)
        >>> features_df = extract_features_batch(results)
        >>> print(features_df[['sensor_id', 'hour', 'ptp', 'freq_1']])
    """
    extractor = FeatureExtractor(sampling_rate=sampling_rate)
    batch_extractor = BatchFeatureExtractor(extractor)
    
    features_list = batch_extractor.extract_from_zip_results(results_list, sensor_type)
    features_df = batch_extractor.to_dataframe(features_list)
    
    return features_df


if __name__ == "__main__":
    """
    테스트 코드
    """
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <bin_file_path>")
        print("\nExample:")
        print("  python feature_extraction.py data/smartcs_1_DNA21006_2024032620.bin")
        sys.exit(1)
    
    # bin 파일 읽기
    from .io_utils import get_df_from_bin
    
    bin_file = Path(sys.argv[1])
    df = get_df_from_bin(bin_file)
    
    print(f"\n{'='*60}")
    print(f"Feature Extraction Test")
    print(f"{'='*60}")
    print(f"File: {bin_file.name}")
    print(f"Records: {len(df):,}")
    print(f"{'='*60}\n")
    
    # 특성값 추출
    extractor = FeatureExtractor(sampling_rate=100)
    features = extractor.extract_all(df)
    
    # 결과 출력
    print("Extracted Features:")
    print(f"{'='*60}")
    
    print("\n[Basic Statistics]")
    print(f"  Mean:   {features['mean']:.6f} m/s²")
    print(f"  Std:    {features['std']:.6f} m/s²")
    print(f"  Min:    {features['min']:.6f} m/s²")
    print(f"  Max:    {features['max']:.6f} m/s²")
    print(f"  Median: {features['median']:.6f} m/s²")
    
    print("\n[Amplitude Features]")
    print(f"  PTP:    {features['ptp']:.6f} m/s²")
    print(f"  RMS:    {features['rms']:.6f} m/s²")
    
    print("\n[Noise]")
    print(f"  Noise Level: {features['noise_level']:.6f} m/s²")
    
    print("\n[Natural Frequencies (FFT)]")
    if features['natural_freqs']:
        for i, freq in enumerate(features['natural_freqs'], 1):
            print(f"  {i}차 모드: {freq:.3f} Hz")
    else:
        print("  No dominant frequencies detected")
    
    print("\n[Shape Features]")
    print(f"  Kurtosis:  {features['kurtosis']:.4f}")
    print(f"  Skewness:  {features['skewness']:.4f}")
    
    print("\n[Validation]")
    print(f"  Valid: {features['is_valid']}")
    if features['validation_issues']:
        print(f"  Issues: {', '.join(features['validation_issues'])}")
    
    print(f"\n{'='*60}")