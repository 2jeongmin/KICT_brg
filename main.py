import sys
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.signal import find_peaks, peak_widths
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# --- 프로젝트 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Module import ---
try:
    from utils.zip_handler import ZipHandler
    from utils.feature_extraction import FeatureExtractor
    from utils.io_utils import parse_bin_filename
except ImportError as e:
    logging.error(f"모듈 임포트 실패: {e}")
    logging.error("프로젝트 루트에 'utils' 폴더와 '__init__.py'가 있는지 확인하십시오.")
    sys.exit(1)

# --- 사용자 설정 ---

# 폴더 경로 설정
BASE_DATA_DIR = Path(r"/media/user/My Book/027. KICT_BMAPS/upload_backup")

# 출력 결과 폴더 경로 설정
BASE_OUTPUT_DIR = Path(r"/home/user/WindowsShare/05. Data/01. Under_Process/027. KICT_BMAPS/upload_backup_results")

# FFT 분석 매개변수 설정
SAMPLING_RATE = 100  # [Hz]
PLOT_FREQ_MIN = 0.5  # [Hz]
PLOT_FREQ_MAX = 30.0 # [Hz]

# 적응형 임계값 설정에 필요한 목표 모드 개수
TARGET_MIN_MODES = 1
TARGET_MAX_MODES = 5

# FFT 플롯 설정
MASTER_FFT_RESOLUTION = 2000

# 병렬 처리 설정
N_WORKERS_SENSORS = 8  # 센서 폴더 레벨 병렬화 (CPU 코어 수에 맞게 조정)

def setup_logging():
    """기본 로깅 설정 - 콘솔 및 파일 동시 출력"""
    from datetime import datetime

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 로그 파일명 (타임스탬프 포함)
    log_filename = BASE_OUTPUT_DIR / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 기본 로거 설정
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷터 생성
    formatter = logging.Formatter('%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s')
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logging.info(f"로그 파일: {log_filename}")
    logging.info("="*80)

def classify_sensor_type(sensor_id: str) -> str:
    """
    센서 ID로부터 센서 타입 판별
    
    Returns:
        'acceleration': 명확한 가속도 센서 (FFT 분석)
        'static': 명확한 정적 센서 (기본 통계량만)
        'unknown': 알 수 없음 (가속도 시도 후 실패시 정적으로 폴백)
    
    판별 규칙:
        1. GW 포함 → 명확한 정적 센서
        2. DNA로 시작 (GW 제외) → 명확한 가속도 센서  
        3. 그 외 (SKYB, GCSKYB 등) → unknown (시도 후 결정)
    """
    sensor_upper = sensor_id.upper()
    
    # 1. 명확한 정적 센서: GW 포함 (DNAGW, GW 등)
    if 'GW' in sensor_upper:
        return 'static'
    
    # 2. 명확한 가속도 센서: DNA로 시작
    if sensor_id.startswith('DNA'):
        return 'acceleration'
    
    # 3. 그 외 → 시도 후 결정
    return 'unknown'

def find_peaks_iterative(spectrum, target_min=1, target_max=5, max_iter=15):
    """
    목표 개수의 피크를 찾기 위해 돌출부(prominence) 임계값을 반복적으로 조정
    """
    current_prominence = (np.max(spectrum) - np.min(spectrum)) * 0.05
    step_up_ratio = 1.5
    step_down_ratio = 0.7
    
    best_peaks = []
    best_props = {}
    best_count_diff = float('inf')
    
    for i in range(max_iter):
        peaks, props = find_peaks(spectrum, prominence=current_prominence)
        count = len(peaks)
        
        if target_min <= count <= target_max:
            return peaks, props, current_prominence
        
        diff = min(abs(count - target_min), abs(count - target_max))
        if diff < best_count_diff:
            best_count_diff = diff
            best_peaks = peaks
            best_props = props
        
        if count > target_max:
            current_prominence *= step_up_ratio
        elif count < target_min:
            if current_prominence < 1e-6:
                break
            current_prominence *= step_down_ratio
            
    return best_peaks, best_props, current_prominence

def get_half_prom_widths_accurate(spectrum, xf, peaks, properties):
    """
    Half-prominence width를 정확히 계산하고 실제 교차점 주파수 반환
    
    Args:
        spectrum: FFT 스펙트럼 진폭
        xf: 주파수 배열 [Hz]
        peaks: 피크 인덱스
        properties: find_peaks에서 반환한 properties
    
    Returns:
        widths_hz: 각 피크의 width [Hz]
        left_freqs: 왼쪽 교차점 주파수 [Hz]
        right_freqs: 오른쪽 교차점 주파수 [Hz]
    """
    try:
        prominence_data = (
            properties['prominences'], 
            properties['left_bases'], 
            properties['right_bases']
        )
        # peak_widths는 4개 값 반환:
        # - widths: width in samples
        # - width_heights: height at which widths are measured
        # - left_ips: left interpolated positions
        # - right_ips: right interpolated positions
        widths_idx, width_heights, left_ips, right_ips = peak_widths(
            spectrum, peaks, rel_height=0.5, prominence_data=prominence_data
        )
        
        # 인덱스를 주파수로 변환
        # xf가 균일 간격이라고 가정
        freq_resolution = xf[1] - xf[0]
        left_freqs = xf[0] + left_ips * freq_resolution
        right_freqs = xf[0] + right_ips * freq_resolution
        widths_hz = (right_ips - left_ips) * freq_resolution
        
        return widths_hz, left_freqs, right_freqs
        
    except Exception as e:
        logging.error(f"peak_widths 계산 실패: {e}")
        # 폴백: prominence_data 없이 계산
        widths_idx, width_heights, left_ips, right_ips = peak_widths(
            spectrum, peaks, rel_height=0.5
        )
        freq_resolution = xf[1] - xf[0]
        left_freqs = xf[0] + left_ips * freq_resolution
        right_freqs = xf[0] + right_ips * freq_resolution
        widths_hz = (right_ips - left_ips) * freq_resolution
        return widths_hz, left_freqs, right_freqs

def filter_close_peaks_adaptive(freqs, amps, widths_hz, peak_indices, spectrum_resolution_hz):
    if len(freqs) == 0:
        return []
        
    candidates = []
    for i in range(len(freqs)):
        candidates.append({
            'freq': freqs[i],
            'amp': amps[i],
            'width': widths_hz[i],
            'idx': peak_indices[i]
        })
    
    while True:
        candidates.sort(key=lambda x: x['freq'])
        merged_happened = False
        
        if len(candidates) < 2:
            break
            
        for i in range(len(candidates) - 1):
            left = candidates[i]
            right = candidates[i+1]
            
            distance = right['freq'] - left['freq']
            avg_width = (left['width'] + right['width']) / 2.0
            
            if distance < avg_width:
                if left['amp'] >= right['amp']:
                    candidates.pop(i+1)
                else:
                    candidates.pop(i)
                merged_happened = True
                break
        
        if not merged_happened:
            break
            
    return candidates

def plot_average_fft(ax: plt.Axes, 
                     xf_master: np.ndarray, 
                     mean_power: np.ndarray, 
                     p05_power: np.ndarray, 
                     p95_power: np.ndarray, 
                     peak_indices: np.ndarray, 
                     peak_prominences: np.ndarray,
                     peak_left_freqs: np.ndarray,
                     peak_right_freqs: np.ndarray,
                     title_info: Dict,
                     show_peaks: bool = True):


    # 평균 스펙트럼 (파란색 라인)
    ax.plot(xf_master, mean_power, color='blue', linewidth=2.0, label='Average amplitude')
    
    # 범위 (파란색 반투명 영역)
    ax.fill_between(xf_master, p05_power, p95_power, color='blue', alpha=0.2, label='Amplitude range')
    
    if show_peaks and peak_indices.size > 0:
        peak_freqs = xf_master[peak_indices]
        peak_amps = mean_power[peak_indices]
        
        # 피크 마커
        ax.scatter(peak_freqs, peak_amps, marker='x', c='red', s=200, 
                   linewidths=3, zorder=5, label='Identified modes')
        
        # prominence 표시
        for i, (pf, pa, prom) in enumerate(zip(peak_freqs, peak_amps, peak_prominences)):
            if pf > 0 and prom > 0:
                ax.annotate('', xy=(pf, pa), xytext=(pf, pa - prom),
                           arrowprops=dict(arrowstyle='-', color='orange', lw=2))
                
                if i == 0:
                    ax.plot([pf, pf], [pa - prom, pa], color='orange', lw=2, 
                           label='Prominence', zorder=4)
        
        # FWHP (Full Width at Half-Prominence)
        for i, (pf, pa) in enumerate(zip(peak_freqs, peak_amps)):
            if pf > 0 and peak_prominences[i] > 0:
                half_height = pa - peak_prominences[i] / 2.0
                left_f = peak_left_freqs[i]   # 실제 스펙트럼과 교차하는 왼쪽 주파수
                right_f = peak_right_freqs[i] # 실제 스펙트럼과 교차하는 오른쪽 주파수
                
                if i == 0:
                    ax.plot([left_f, right_f], [half_height, half_height], 
                        color='purple', lw=2, label='FWHP', zorder=4)
                else:
                    ax.plot([left_f, right_f], [half_height, half_height], 
                        color='purple', lw=2, zorder=4)

        # 주파수 텍스트
        for pf, pa in zip(peak_freqs, peak_amps):
            if pf > 0:
                ax.text(pf, pa*1.05, f'{pf:.3f} Hz', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='black', alpha=0.8))
    
    ax.set_xlabel('Frequency [Hz]', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.set_title(
        f"Sensor: {title_info['sensor_id']}\n"
        f"Period: {title_info['start_time']} ~ {title_info['end_time']} ({title_info['file_count']} files)",
        fontsize=12
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

def plot_frequency_trend(ax: plt.Axes, 
                         freq_trend_df: pd.DataFrame, 
                         identified_modes: np.ndarray,
                         title_info: Dict):
    """시간에 따른 주파수 변화 추세를 표시"""
    
    if freq_trend_df.empty:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', fontsize=14)
        return
    
    mode_columns = ['1st_mode', '2nd_mode', '3rd_mode']
    colors = ['red', 'green', 'blue']
    
    for i, (col, color, ref_freq) in enumerate(zip(mode_columns, colors, identified_modes)):
        if col not in freq_trend_df.columns:
            continue
        
        valid_data = freq_trend_df[freq_trend_df[col] > 0].copy()
        if valid_data.empty:
            continue
        
        ax.plot(valid_data['time'], valid_data[col], 
               color=color, linewidth=1.5, marker='o', markersize=3,
               label=f'Mode {i+1} (ref: {ref_freq:.2f} Hz)')
        
        if ref_freq > 0:
            ax.axhline(y=ref_freq, color=color, linestyle='--', alpha=0.5, linewidth=1.0)
    
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Frequency [Hz]', fontsize=14)
    ax.set_title(
        f"Sensor: {title_info['sensor_id']}\n"
        f"Period: {title_info['start_time']} ~ {title_info['end_time']}",
        fontsize=12
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.tick_params(axis='x', rotation=45)

def process_single_zip_for_pass1(args):
    """
    PASS 1: 단일 zip 파일을 처리하여 시간당 FFT 데이터를 추출
    """
    zip_path, sensor_id = args
    handler = ZipHandler(mode='stream')
    extractor = FeatureExtractor(
        sampling_rate=SAMPLING_RATE,
        fft_freq_range=(PLOT_FREQ_MIN, PLOT_FREQ_MAX),
        fft_n_peaks=10
    )
    
    xf_master = np.linspace(PLOT_FREQ_MIN, PLOT_FREQ_MAX, MASTER_FFT_RESOLUTION)
    
    try:
        bin_files_in_zip = handler.list_bin_files_in_zip(zip_path)
        if not bin_files_in_zip:
            return []
        
        results = []
        for bin_filename in bin_files_in_zip:
            try:
                df = handler.get_df_from_zip(zip_path, bin_filename)
                if df is None or df.empty:
                    continue
                
                metadata = parse_bin_filename(Path(bin_filename))
                hour = metadata['hour']
                
                try:
                    xf_orig, power_orig = extractor.get_fft_spectrum(df['value'].values)
                except Exception as e:
                    logging.error(f"FFT failed for {bin_filename}: {e}")
                    continue
                
                interpolated_power = np.interp(xf_master, xf_orig, power_orig)
                results.append((interpolated_power, hour, df, xf_orig, power_orig))
                
            except Exception as e:
                logging.error(f"Error in {bin_filename}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logging.error(f"Bad zip file {zip_path.name}: {e}")
        return []

def find_hourly_peaks_in_bins(hour, df, extractor, search_bins, mode_column_names, sensor_id):
    """
    PASS 2: 개별 시간별 데이터에서 탐색 빈 내의 피크를 찾음
    """
    # 기본 데이터 구조
    hourly_peaks = {
        'time': hour,
        'sensor_id': sensor_id,
        'ptp': 0.0,
        'mean': 0.0,
        'std': 0.0,
        'rms': 0.0,
        'noise_level': None,  # 다른 알고리즘으로 처리
        'data_count': len(df),
        'is_valid': True
    }
    
    # 고유 모드 추가
    for col in mode_column_names:
        hourly_peaks[col] = 0.0
    
    try:
        values = df['value'].values
        
        # 1. 기본 통계도 계산
        hourly_peaks['ptp'] = float(np.ptp(values))
        hourly_peaks['mean'] = float(np.mean(values))
        hourly_peaks['std'] = float(np.std(values))
        hourly_peaks['rms'] = float(np.sqrt(np.mean(values**2)))
        
        # 2. 간단한 데이터 검증
        hourly_peaks['is_valid'] = (
            np.all(np.isfinite(values)) and 
            len(values) < 400000 and
            -100 < np.min(values) and np.max(values) < 100
        )
        
        # 3. FFT 기반 고유 진동수 추출
        xf, power = extractor.get_fft_spectrum(values)
        
        for i, (bin_min, bin_max) in enumerate(search_bins):
            col_name = mode_column_names[i]
            
            if bin_min == 0.0 and bin_max == 0.0:
                hourly_peaks[col_name] = 0.0
                continue
            
            mask = (xf >= bin_min) & (xf <= bin_max)
            xf_bin = xf[mask]
            power_bin = power[mask]
            
            if len(xf_bin) == 0:
                hourly_peaks[col_name] = 0.0
                continue
            
            peak_idx = np.argmax(power_bin)
            hourly_peaks[col_name] = xf_bin[peak_idx]
    
    except Exception as e:
        logging.warning(f"[{sensor_id}] {hour} 처리 실패: {e}")
        hourly_peaks['is_valid'] = False
    
    return hourly_peaks

def process_static_sensor_folder(sensor_dir: Path, base_output_dir: Path):
    """정적 센서 처리 (온도, 습도, 변위, 균열 등)"""
    # from utils.database import TimeseriesDB
    
    sensor_id = sensor_dir.name
    logging.info(f"===== STATIC 센서 [{sensor_id}] 처리 중 =====")
    
    sensor_output_dir = base_output_dir / sensor_id
    csv_path = sensor_output_dir / f"{sensor_id}_static_features.csv"

    if csv_path.exists():
        logging.info(f"[{sensor_id}] 정적 센서 결과 파일이 이미 존재합니다.")
        return

    # db = TimeseriesDB()
    
    # if db.check_sensor_exists(sensor_id):
    #     logging.info(f"[{sensor_id}] 이미 처리됨 - 건너뜀")
    #     return
    
    handler = ZipHandler(mode='stream')
    extractor = FeatureExtractor(
        sampling_rate=SAMPLING_RATE,
        fft_freq_range=(PLOT_FREQ_MIN, PLOT_FREQ_MAX),
        fft_n_peaks=10
    )
    
    zip_files = handler.find_zip_files(sensor_dir, recursive=False)
    if not zip_files:
        raise FileNotFoundError(f"{sensor_dir}에서 zip 파일을 찾을 수 없습니다.")
    
    all_hourly_records = []
    
    for zip_path in tqdm(zip_files, desc=f"[{sensor_id}] 정적 센서 처리"):
        try:
            bin_files_in_zip = handler.list_bin_files_in_zip(zip_path)
            if not bin_files_in_zip:
                continue
            
            for bin_filename in bin_files_in_zip:
                try:
                    df = handler.get_df_from_zip(zip_path, bin_filename)
                    if df is None or df.empty:
                        continue
                    
                    metadata = parse_bin_filename(Path(bin_filename))
                    hour = metadata['hour']
                    
                    features = extractor.extract_all(df, sensor_type='static')

                    record = {
                        'time': hour,
                        'sensor_id': sensor_id,
                        'mean': features.get('mean', 0.0),
                        'std': features.get('noise_level', 0.0),
                        'min': features.get('min', 0.0),
                        'max': features.get('max', 0.0),
                        'ptp': features.get('ptp', 0.0),
                        'data_count': features.get('data_count', len(df)),
                        'is_valid': features.get('is_valid', True)
                    }
                    all_hourly_records.append(record)
                    
                except Exception as e:
                    logging.error(f"Error processing {bin_filename}: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Bad zip file {zip_path.name}: {e}")
            continue
    
    if all_hourly_records:
        # CSV로 저장
        output_dir = base_output_dir / sensor_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(all_hourly_records)
        csv_path = output_dir / f"{sensor_id}_static_features.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"[{sensor_id}] {len(all_hourly_records)}개의 시간별 기록을 저장했습니다: {csv_path}")
    else:
        logging.warning(f"[{sensor_id}] 처리된 데이터가 없습니다.")

def process_sensor_folder(sensor_dir: Path, base_output_dir: Path):
    """
    가속도 센서 처리 (FFT 분석 포함)
    2-pass 접근법:
        Pass 1: 모든 파일의 FFT를 계산하여 평균 스펙트럼 생성 및 고유 모드 식별
        Pass 2: 식별된 모드를 기반으로 각 시간별 파일의 주파수 추출
    """
    sensor_id = sensor_dir.name
    logging.info(f"===== 센서 [{sensor_id}] 처리 중 =====")
    
    sensor_output_dir = base_output_dir / sensor_id
    sensor_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미 처리된 파일이 있는지 확인
    TARGET_N_MODES = 3
    expected_files = [
        sensor_output_dir / f"{sensor_id}_dynamic_features.csv",
        sensor_output_dir / f"{sensor_id}_1_avg_fft_with_peaks.png",
        sensor_output_dir / f"{sensor_id}_3_freq_trend_{TARGET_N_MODES}_modes.png"
    ]
    
    if all(f.exists() for f in expected_files):
        logging.info(f"[{sensor_id}] 이미 처리됨 - 건너뜀")
        return

    handler = ZipHandler(mode='stream')
    extractor = FeatureExtractor(
        sampling_rate=SAMPLING_RATE,
        fft_freq_range=(PLOT_FREQ_MIN, PLOT_FREQ_MAX),
        fft_n_peaks=10
    )

    zip_files = handler.find_zip_files(sensor_dir, recursive=False)
    if not zip_files:
        raise FileNotFoundError(f"{sensor_dir}에서 zip 파일을 찾을 수 없습니다.")

    # --- PASS 1 ---
    all_interpolated_powers = []
    hourly_fft_data_cache = []
    xf_master = np.linspace(PLOT_FREQ_MIN, PLOT_FREQ_MAX, MASTER_FFT_RESOLUTION)
    freq_resolution = (PLOT_FREQ_MAX - PLOT_FREQ_MIN) / (MASTER_FFT_RESOLUTION - 1)

    zip_args = [(zip_path, sensor_id) for zip_path in zip_files]
    
    # [FIX] 중첩 병렬 처리 제거: 순차 처리
    # 이미 상위 레벨에서 센서 단위 병렬화가 되어 있으므로 여기서는 Loop 사용
    all_results = []
    for args in tqdm(zip_args, desc=f"Pass 1/2 [{sensor_id}]"):
        res = process_single_zip_for_pass1(args)
        all_results.append(res)
    
    for file_results in all_results:
        for interpolated_power, hour, df, xf_orig, power_orig in file_results:
            hourly_fft_data_cache.append((hour, df, xf_orig, power_orig))
            if interpolated_power is not None:
                all_interpolated_powers.append(interpolated_power)
    
    if not all_interpolated_powers:
        raise ValueError(f"센서 {sensor_id}에 대해 유효한 FFT 데이터가 발견되지 않았습니다.")

    # 평균 및 백분위수 스펙트럼 계산
    power_matrix = np.array(all_interpolated_powers)
    mean_power = np.mean(power_matrix, axis=0)
    p05_power = np.percentile(power_matrix, 5, axis=0)
    p95_power = np.percentile(power_matrix, 95, axis=0)

    # 1. 적응형 피크 감지
    peaks_idx, properties, final_threshold = find_peaks_iterative(
        mean_power, 
        target_min=TARGET_MIN_MODES, 
        target_max=TARGET_MAX_MODES
    )
    logging.info(f"[{sensor_id}] 반복 감지: {len(peaks_idx)}개의 피크 발견 (임계값: {final_threshold:.4f})")

    if peaks_idx.size == 0:
        raise ValueError(f"센서 {sensor_id}에 대해 구조적 모드가 감지되지 않았습니다.")

    # 2. 반-돌출부 너비 계산
    prominences = properties['prominences']
    widths_hz, left_freqs, right_freqs = get_half_prom_widths_accurate(
        mean_power, xf_master, peaks_idx, properties
    )
    
    peak_freqs = xf_master[peaks_idx]
    peak_amps = mean_power[peaks_idx]

    # 3. 적응형 모드 분리
    filtered_candidates = filter_close_peaks_adaptive(
        peak_freqs, peak_amps, widths_hz, peaks_idx, freq_resolution
    )
    
    # 주파수 오름차순 정렬
    filtered_candidates.sort(key=lambda x: x['freq'])
    identified_modes_hz = np.array([c['freq'] for c in filtered_candidates])
    identified_indices = np.array([c['idx'] for c in filtered_candidates])
    
    # 3개 모드
    TARGET_N_MODES = 3
    
    if len(filtered_candidates) > TARGET_N_MODES:
        # 4개 이상 → 가중 스코어로 선택 (저주파 우선 + 진폭 고려)
        logging.info(f"[{sensor_id}] {len(filtered_candidates)}개 모드 발견 → 가중 스코어로 {TARGET_N_MODES}개 선택")
        
        # 스코어 계산: 저주파 70% + 진폭 30%
        all_freqs = np.array([c['freq'] for c in filtered_candidates])
        all_amps = np.array([c['amp'] for c in filtered_candidates])
        
        for c in filtered_candidates:
            # 저주파 가중: 1차 모드(~5Hz)가 고차(~15Hz)보다 3배 선호
            freq_weight = np.exp(-c['freq'] / 10.0)
            # 진폭 가중: 정규화
            amp_normalized = c['amp'] / np.max(all_amps)
            # 최종 스코어: 저주파 70%, 진폭 30%
            c['score'] = 0.7 * freq_weight + 0.3 * amp_normalized
        
        # 스코어 높은 순 정렬 후 상위 3개 선택
        filtered_candidates.sort(key=lambda x: x['score'], reverse=True)
        selected = filtered_candidates[:TARGET_N_MODES]
        
        # 다시 주파수 오름차순으로 정렬 (1차, 2차, 3차 순서 유지)
        selected.sort(key=lambda x: x['freq'])
        filtered_candidates = selected
        
        identified_modes_hz = np.array([c['freq'] for c in filtered_candidates])
        identified_indices = np.array([c['idx'] for c in filtered_candidates])
        
    elif len(filtered_candidates) < TARGET_N_MODES:
        # 1~2개 → 부족한 차수는 0Hz
        n_missing = TARGET_N_MODES - len(filtered_candidates)
        logging.info(f"[{sensor_id}] {len(filtered_candidates)}개 모드만 발견 → {n_missing}개 차수는 0Hz 패딩")
        identified_modes_hz = np.concatenate([identified_modes_hz, np.zeros(n_missing)])
        for _ in range(n_missing):
            filtered_candidates.append({'freq': 0.0, 'amp': 0.0, 'width': 0.0, 'idx': -1})
    
    # 3개 확인
    assert len(filtered_candidates) == TARGET_N_MODES
    assert len(identified_modes_hz) == TARGET_N_MODES
    
    # identified_prominences 재구성
    identified_prominences = []
    original_idx_list = list(peaks_idx)
    for c in filtered_candidates:
        if c['idx'] in original_idx_list:
            pos = original_idx_list.index(c['idx'])
            identified_prominences.append(prominences[pos])
        else:
            identified_prominences.append(0)
    identified_prominences = np.array(identified_prominences)

    # identified_widths_hz 재구성 (FWHP 시각화용)
    identified_left_freqs = []
    identified_right_freqs = []
    for c in filtered_candidates:
        if c['idx'] in original_idx_list:
            pos = original_idx_list.index(c['idx'])
            identified_left_freqs.append(left_freqs[pos])
            identified_right_freqs.append(right_freqs[pos])
        else:
            identified_left_freqs.append(0)
            identified_right_freqs.append(0)
    identified_left_freqs = np.array(identified_left_freqs)
    identified_right_freqs = np.array(identified_right_freqs)

    logging.info(f"[{sensor_id}] 최종 고유 모드 (Hz): {np.round(identified_modes_hz, 3)}")

    # 4. 탐색 빈 정의
    search_bins = []
    for i, candidate in enumerate(filtered_candidates):
        if candidate['freq'] == 0.0:
            search_bins.append((0.0, 0.0))
            continue
        # 실제 교차점을 탐색 범위로 사용
        bin_min = max(PLOT_FREQ_MIN, identified_left_freqs[i])
        bin_max = min(PLOT_FREQ_MAX, identified_right_freqs[i])
        search_bins.append((bin_min, bin_max))

    logging.info(f"[{sensor_id}] 탐색 (반-돌출부 너비): {[(round(a,2), round(b,2)) for a,b in search_bins]}")
    mode_column_names = ['1st_mode', '2nd_mode', '3rd_mode']

    # --- PASS 2 ---
    hourly_freq_data_for_csv = []
    for hour, df, xf_orig, power_orig in tqdm(hourly_fft_data_cache, desc=f"Pass 2/2 [{sensor_id}]"):
        if df is None: 
            empty_data = {
                'time': hour,
                'sensor_id': sensor_id,
                'ptp': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'rms': 0.0,
                'noise_level': None,
                'data_count': 0,
                'is_valid': False
            }
            for col in mode_column_names: empty_data[col] = 0.0
            hourly_freq_data_for_csv.append(empty_data)
            continue
        hourly_peaks = find_hourly_peaks_in_bins(hour, df, extractor, search_bins, mode_column_names, sensor_id)
        hourly_freq_data_for_csv.append(hourly_peaks)

    # 결과 저장
    try:
        start_time_str = parse_bin_filename(zip_files[0])['hour'].strftime('%Y-%m-%d %H:00')
        end_time_str = parse_bin_filename(zip_files[-1])['hour'].strftime('%Y-%m-%d %H:00')
    except:
        start_time_str = "N/A"
        end_time_str = "N/A"

    title_info = {
        'sensor_id': sensor_id,
        'start_time': start_time_str,
        'end_time': end_time_str,
        'file_count': len(all_interpolated_powers)
    }

    freq_trend_df = pd.DataFrame(hourly_freq_data_for_csv)
    if not freq_trend_df.empty:
        freq_trend_df = freq_trend_df.sort_values(by='time')
        csv_filename = sensor_output_dir / f"{sensor_id}_dynamic_features.csv"
        freq_trend_df.to_csv(csv_filename, index=False)

    try:
        fig1, ax1 = plt.subplots(figsize=(16, 8))
        plot_average_fft(ax1, xf_master, mean_power, p05_power, p95_power, 
                        identified_indices, identified_prominences,
                        identified_left_freqs, identified_right_freqs,
                        title_info, show_peaks=True)
        fig1.suptitle(f"Sensor Analysis : {sensor_id}", fontsize=16, fontweight='bold')
        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig1.savefig(sensor_output_dir / f"{sensor_id}_1_avg_fft_with_peaks.png", dpi=150)
        plt.close(fig1)
    except Exception as e:
        logging.error(f"Plot 1 Error: {e}")

    try:
        fig3, ax3 = plt.subplots(figsize=(16, 8))
        plot_frequency_trend(ax3, freq_trend_df, identified_modes_hz, title_info) 
        fig3.suptitle(f"Frequency Trend : {sensor_id}", fontsize=16, fontweight='bold')
        fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig3.savefig(sensor_output_dir / f"{sensor_id}_3_freq_trend_{TARGET_N_MODES}_modes.png", dpi=150)
        plt.close(fig3)
    except Exception as e:
        logging.error(f"Plot 3 Error: {e}")

def process_sensor_wrapper(args):
    sensor_dir, base_output_dir = args
    sensor_id = sensor_dir.name
    sensor_type = classify_sensor_type(sensor_id)
    
    try:
        if sensor_type == 'acceleration':
            logging.info(f"[{sensor_id}] ACCELERATION 센서로 감지됨")
            try:
                process_sensor_folder(sensor_dir, base_output_dir)
                logging.info(f"[{sensor_id}] ACCELERATION 처리 완료")
            except Exception as acc_error:
                logging.error(f"[{sensor_id}] ACCELERATION 처리 실패: {acc_error}")
                raise
            
        elif sensor_type == 'static':
            logging.info(f"[{sensor_id}] STATIC 센서로 감지됨")
            process_static_sensor_folder(sensor_dir, base_output_dir)
            
        else:
            logging.info(f"[{sensor_id}] 알 수 없는 유형 - ACCELERATION 먼저 시도")
            try:
                process_sensor_folder(sensor_dir, base_output_dir)
                logging.info(f"[{sensor_id}] ACCELERATION으로 성공적으로 처리됨")
            except Exception as acc_error:
                logging.warning(f"[{sensor_id}] ACCELERATION 실패: {acc_error}")
                logging.info(f"[{sensor_id}] STATIC 처리로 폴백")
                process_static_sensor_folder(sensor_dir, base_output_dir)
                
        return sensor_id, True
    except Exception as e:
        logging.error(f"Fatal error [{sensor_id}]: {e}")
        return sensor_id, False

def main():
    setup_logging()

    sensor_dirs = [
        d for d in BASE_DATA_DIR.iterdir() 
        if d.is_dir() 
        and not d.name.startswith('.')
        and 'analysis_results' not in d.name
    ]
    
    if not sensor_dirs:
        logging.error("BASE_DATA_DIR에서 유효한 센서 디렉토리를 찾을 수 없습니다.")
        return
    
    logging.info(f"총 {len(sensor_dirs)}개의 센서 폴더를 발견했습니다.")
    
    sensor_args = [(d, BASE_OUTPUT_DIR) for d in sensor_dirs]
    
    # 센서 레벨에서 병렬 처리
    with Pool(N_WORKERS_SENSORS) as pool:
        list(tqdm(pool.imap(process_sensor_wrapper, sensor_args), total=len(sensor_dirs), desc="센서 처리 중"))

if __name__ == "__main__":
    main()
