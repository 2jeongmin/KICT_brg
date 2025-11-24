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
BASE_DATA_DIR = Path(r"/home/user/WindowsShare/05. Data/01. Under_Process/027. KICT_BMAPS/upload")

# 출력 결과 폴더 경로 설정
BASE_OUTPUT_DIR = Path(r"/home/user/WindowsShare/05. Data/01. Under_Process/027. KICT_BMAPS/upload") / "analysis_results_parallel_adaptive"

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
N_WORKERS_SENSORS = 16  # 센서 폴더 레벨 병렬화 (CPU 코어 수에 맞게 조정)
N_WORKERS_ZIP_FILES = 6  # 각 센서 내 zip 파일 병렬화 (CPU 코어 수에 맞게 조정)

# ---

def setup_logging():
    """기본 로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s'
    )

def classify_sensor_type(sensor_id: str) -> str:
    """
    센서 ID로부터 센서 타입 판별
    
    Returns:
        'acceleration': 가속도 센서 (FFT 분석)
        'static': 정적 센서 (기본 통계량만)
        'unknown': 알 수 없음 (가속도 시도 후 실패시 정적으로 폴백)
    
    판별 규칙:
        1. DNAGW 포함 → 정적 센서
        2. DNA로 시작 (DNAGW 제외) → 가속도 센서
        3. 그 외 → unknown (시도 후 결정)
    """
    sensor_upper = sensor_id.upper()
    
    # DNAGW 정적 센서
    if 'DNAGW' in sensor_upper:
        return 'static'
    
    # DNA로 시작하면 가속도 센서
    if sensor_id.startswith('DNA'):
        return 'acceleration'
    
    # 알 수 없는 경우 - 가속도로 시도 후 결정
    return 'unknown'

def find_peaks_iterative(spectrum, target_min=1, target_max=5, max_iter=15):
    """
    목표 개수의 피크를 찾기 위해 돌출부(prominence) 임계값을 반복적으로 조정
    """
    # 초기 추정값: 범위의 5%
    current_prominence = (np.max(spectrum) - np.min(spectrum)) * 0.05
    
    step_up_ratio = 1.5
    step_down_ratio = 0.7
    
    best_peaks = []
    best_props = {}
    best_count_diff = float('inf')
    
    for i in range(max_iter):
        peaks, props = find_peaks(spectrum, prominence=current_prominence)
        count = len(peaks)
        
        # 목표 범위 내에 있는지 확인
        if target_min <= count <= target_max:
            return peaks, props, current_prominence
        
        # 실패할 경우를 대비하여 가장 근접한 결과를 추적
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

def get_half_prom_widths(spectrum, peaks, prominences):
    """
    주어진 피크에 대한 돌출부 반치전폭(half-prominence) 계산
    너비를 인덱스 단위로 반환
    """
    # peak_widths는 (너비, 너비_높이, 왼쪽_IP, 오른쪽_IP)를 반환
    # rel_height=0.5는 돌출부의 절반을 의미
    widths, _, _, _ = peak_widths(spectrum, peaks, rel_height=0.5, prominence_data=prominences)
    return widths

def filter_close_peaks_adaptive(freqs, amps, widths_hz, peak_indices, spectrum_resolution_hz):
    """
    평균 돌출부 반치전폭보다 가까운 피크들을 병합
    더 높은 진폭을 가진 피크를 우선적으로 유지
    
    Args:
        freqs: 피크의 주파수
        amps: 피크의 진폭
        widths_hz: 피크의 반-돌출부 너비 (Hz 단위)
        peak_indices: 스펙트럼 배열 내 원래 인덱스
        
    Returns:
        필터링된 피크의 튜플 리스트 (freq, amp, width, original_index)
    """
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
            
            # 조건: 거리 < 평균 너비
            if distance < avg_width:
                # 병합: 더 강한 피크 유지
                if left['amp'] >= right['amp']:
                    # 왼쪽 유지, 오른쪽 제거
                    candidates.pop(i+1)
                else:
                    # 오른쪽 유지, 왼쪽 제거
                    candidates.pop(i)
                merged_happened = True
                break # 루프 재시작
        
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
                     title_info: Dict,
                     show_peaks: bool = True):
    """
    (subplot 1) 시각적 돌출부를 포함한 평균 FFT 스펙트럼
    """
    ax.plot(xf_master, mean_power, color='blue', linewidth=2.0, label='평균 진폭')
    ax.fill_between(xf_master, p05_power, p95_power, color='blue', alpha=0.2, label='진폭 범위')
    
    if show_peaks and peak_indices.size > 0:
        peak_freqs = xf_master[peak_indices]
        peak_powers = mean_power[peak_indices]
        
        ax.plot(peak_freqs, peak_powers, 'rx', markersize=8, mew=2, label=f'감지된 모드 ({len(peak_freqs)})', zorder=5)
        
        # 수직 돌출부 선
        peak_bases = peak_powers - peak_prominences
        ax.vlines(x=peak_freqs, ymin=peak_bases, ymax=peak_powers, color='orange', linewidth=2, label='돌출부', zorder=4)

        for freq, power in zip(peak_freqs, peak_powers):
            annotation_text = f"{freq:.3f} Hz"
            ax.annotate(
                annotation_text, (freq, power),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=12, fontweight='bold', color='black',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="k", lw=0.5, alpha=0.8),
                zorder=6
            )

    title = f"Data Period: {title_info['start_time']} to {title_info['end_time']} (Total {title_info['file_count']} files)"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.set_xlim(PLOT_FREQ_MIN, PLOT_FREQ_MAX)
    if p95_power.size > 0:
        ax.set_ylim(bottom=0, top=np.max(p95_power) * 1.1) 
    else:
        ax.set_ylim(bottom=0)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

def plot_frequency_trend(ax: plt.Axes, df: pd.DataFrame, identified_modes_hz: List[float], title_info: Dict):
    """
    (subplot 2) 식별된 모든 N개의 지배적인 주파수 추세
    """
    mode_columns = [f"mode_{freq:.3f}Hz" for freq in identified_modes_hz]
    valid_data_df = df.dropna(subset=mode_columns, how='all')
    
    if valid_data_df.empty:
        logging.warning(f"[{title_info['sensor_id']}] 어떤 모드에 대해서도 유효한 주파수 데이터가 없음. 추세를 그릴 수 없음.")
        ax.text(0.5, 0.5, "유효한 주파수 데이터가 발견되지 않음", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, color='red', fontsize=12)
        return

    colors = plt.cm.get_cmap('tab10', len(mode_columns))
    markers = ['o', 's', '^', 'v', 'D', '*', 'p', 'h']
    
    all_freqs = []
    for i, col in enumerate(mode_columns):
        if col in df.columns:
            valid_data = df.dropna(subset=[col])
            if not valid_data.empty:
                ax.plot(valid_data['time'], valid_data[col], 
                        marker=markers[i % len(markers)], 
                        linestyle='-', 
                        markersize=4,
                        color=colors(i),
                        label=f"모드 {i+1} (~{identified_modes_hz[i]:.2f} Hz)")
                all_freqs.extend(valid_data[col].tolist())
    
    if not all_freqs:
        min_freq, max_freq, padding = PLOT_FREQ_MIN, PLOT_FREQ_MAX, 0.5
    else:
        min_freq = min(all_freqs)
        max_freq = max(all_freqs)
        padding = (max_freq - min_freq) * 0.1
        if padding == 0: padding = 0.5
    
    ax.set_ylim(min_freq - padding, max_freq + padding)
    title = f"Dominant Natural Frequency Trend (All {len(mode_columns)} Identified Modes)"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Timestamp", fontsize=10)
    ax.set_ylabel("Dominant Frequency (Hz)", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=9)

def find_hourly_peaks_in_bins(
    hour: pd.Timestamp, 
    df: pd.DataFrame, 
    extractor: FeatureExtractor, 
    search_bins: List[Tuple[float, float]],
    mode_column_names: List[str]
) -> Dict:
    """
    N개의 사전 정의된 모드 빈 내에서 유의미한 돌출부를 가진 피크를 찾기 위해 한 시간 분량의 데이터를 처리
    유의미한 prominence를 가진 피크가 없으면 0.0 반환
    """
    hourly_data = {'time': hour}
    for col in mode_column_names:
        hourly_data[col] = 0.0  # 기본값 0 (NaN 대신)

    if df is None or df.empty:
        return hourly_data
    
    values = df['value'].values
    xf_orig, power_orig = extractor.get_fft_spectrum(values)
    
    if len(xf_orig) == 0:
        return hourly_data
    
    for i, (bin_min, bin_max) in enumerate(search_bins):
        col_name = mode_column_names[i]
        
        # 더미 bin (freq=0인 모드)은 skip
        if bin_min == 0.0 and bin_max == 0.0:
            hourly_data[col_name] = 0.0
            continue
            
        mask = (xf_orig >= bin_min) & (xf_orig <= bin_max)
        xf_bin = xf_orig[mask]
        power_bin = power_orig[mask]
        
        if power_bin.size == 0:
            hourly_data[col_name] = 0.0
            continue
        
        # Prominence 기반 피크 검출
        max_power = np.max(power_bin)
        if max_power <= 0:
            hourly_data[col_name] = 0.0
            continue
            
        prominence_threshold = max_power * 0.1  # 최대값의 10%
        peaks, props = find_peaks(power_bin, prominence=prominence_threshold)
        
        if len(peaks) > 0:
            # 가장 강한 prominence를 가진 피크 선택
            strongest_idx_in_peaks = np.argmax(props['prominences'])
            strongest_idx_in_bin = peaks[strongest_idx_in_peaks]
            dominant_freq = xf_bin[strongest_idx_in_bin]
            hourly_data[col_name] = dominant_freq
        else:
            # 유의미한 prominence를 가진 피크 없음
            hourly_data[col_name] = 0.0
            
    return hourly_data

def process_single_zip_for_pass1(args):
    """
    단일 ZIP 파일을 처리하여 보간된 FFT 스펙트럼 데이터를 반환 (Pass 1)
    """
    zip_path, sensor_id = args
    handler = ZipHandler(mode='stream')
    extractor = FeatureExtractor(
        sampling_rate=SAMPLING_RATE,
        fft_freq_range=(PLOT_FREQ_MIN, PLOT_FREQ_MAX),
        fft_n_peaks=10
    )
    
    xf_master = np.linspace(PLOT_FREQ_MIN, PLOT_FREQ_MAX, MASTER_FFT_RESOLUTION)
    results_list = []
    
    try:
        results = handler.process_single_zip(zip_path)
        for res in results:
            df = res.get('df')
            hour = res.get('hour')
            if df is None or df.empty:
                results_list.append((None, hour, df, None, None))
                continue
            values = df['value'].values
            xf_orig, power_orig = extractor.get_fft_spectrum(values)
            if len(xf_orig) == 0:
                results_list.append((None, hour, df, xf_orig, power_orig))
                continue
            interpolated_power = np.interp(xf_master, xf_orig, power_orig)
            results_list.append((interpolated_power, hour, df, xf_orig, power_orig))
    except Exception as e:
        logging.error(f"[{sensor_id}] {zip_path.name} 처리 오류: {e}")
        return []
    return results_list

def process_static_sensor_folder(sensor_dir: Path, base_output_dir: Path):
    """
    정적 센서 처리: FFT 없이 기본 통계량만 추출
    """
    sensor_id = sensor_dir.name
    logging.info(f"===== STATIC 센서 [{sensor_id}] 처리 중 =====")
    
    sensor_output_dir = base_output_dir / sensor_id
    sensor_output_dir.mkdir(parents=True, exist_ok=True)

    handler = ZipHandler(mode='stream')

    zip_files = handler.find_zip_files(sensor_dir, recursive=False)
    if not zip_files:
        logging.warning(f"[{sensor_id}] zip 파일이 발견되지 않았습니다.")
        return

    # 시간별 기본 통계량 추출
    hourly_stats_data = []
    
    for zip_path in tqdm(zip_files, desc=f"Static [{sensor_id}]"):
        try:
            results = handler.process_single_zip(zip_path)
            for res in results:
                hour = res.get('hour')
                df = res.get('df')
                
                if df is None or df.empty:
                    stats = {
                        'time': hour,
                        'mean': np.nan,
                        'std': np.nan,
                        'min': np.nan,
                        'max': np.nan,
                        'ptp': np.nan,
                        'rms': np.nan,
                        'count': 0
                    }
                else:
                    values = df['value'].values
                    stats = {
                        'time': hour,
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'ptp': float(np.ptp(values)),
                        'rms': float(np.sqrt(np.mean(values**2))),
                        'count': len(values)
                    }
                hourly_stats_data.append(stats)
        except Exception as e:
            logging.error(f"[{sensor_id}] {zip_path.name} 처리 오류: {e}")
            continue

    # 결과 저장
    if hourly_stats_data:
        stats_df = pd.DataFrame(hourly_stats_data)
        stats_df = stats_df.sort_values(by='time')
        excel_filename = sensor_output_dir / f"{sensor_id}_static_features.xlsx"
        stats_df.to_excel(excel_filename, index=False, sheet_name='Hourly_Stats')
        logging.info(f"[{sensor_id}] {len(stats_df)}개의 시간별 기록을 저장했습니다.")

def process_sensor_folder(sensor_dir: Path, base_output_dir: Path):
    sensor_id = sensor_dir.name
    logging.info(f"===== 센서 [{sensor_id}] 처리 중 =====")
    
    sensor_output_dir = base_output_dir / sensor_id
    sensor_output_dir.mkdir(parents=True, exist_ok=True)

    handler = ZipHandler(mode='stream')
    extractor = FeatureExtractor(
        sampling_rate=SAMPLING_RATE,
        fft_freq_range=(PLOT_FREQ_MIN, PLOT_FREQ_MAX),
        fft_n_peaks=10
    )

    zip_files = handler.find_zip_files(sensor_dir, recursive=False)
    if not zip_files:
        raise FileNotFoundError(f"{sensor_dir}에서 zip 파일을 찾을 수 없습니다.")

    # --- PASS 1 --- (FFT 계산 및 평균/백분위수 계산)
    all_interpolated_powers = []
    hourly_fft_data_cache = []
    xf_master = np.linspace(PLOT_FREQ_MIN, PLOT_FREQ_MAX, MASTER_FFT_RESOLUTION)
    freq_resolution = (PLOT_FREQ_MAX - PLOT_FREQ_MIN) / (MASTER_FFT_RESOLUTION - 1)

    zip_args = [(zip_path, sensor_id) for zip_path in zip_files]
    with Pool(N_WORKERS_ZIP_FILES) as pool:
        all_results = list(tqdm(pool.imap(process_single_zip_for_pass1, zip_args), total=len(zip_files), desc=f"Pass 1/2 [{sensor_id}]"))
    
    for file_results in all_results:
        for interpolated_power, hour, df, xf_orig, power_orig in file_results:
            # Pass 2를 위해 원본 데이터를 캐시
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

    # 1. 적응형 피크 감지 (Adaptive Peak Detection)
    peaks_idx, properties, final_threshold = find_peaks_iterative(
        mean_power, 
        target_min=TARGET_MIN_MODES, 
        target_max=TARGET_MAX_MODES
    )
    logging.info(f"[{sensor_id}] 반복 감지: {len(peaks_idx)}개의 피크 발견 (임계값: {final_threshold:.4f})")

    if peaks_idx.size == 0:
        raise ValueError(f"센서 {sensor_id}에 대해 구조적 모드가 감지되지 않았습니다.")

    # 2. 반-돌출부 너비 계산 (Hz)
    # peak_widths는 '인덱스' 단위로 너비를 반환
    prominences = properties['prominences']
    widths_idx = get_half_prom_widths(mean_power, peaks_idx, prominences)
    widths_hz = widths_idx * freq_resolution
    
    peak_freqs = xf_master[peaks_idx]
    peak_amps = mean_power[peaks_idx]

    # 3. 적응형 모드 분리 (거리 < 평균 너비인 경우 병합)
    filtered_candidates = filter_close_peaks_adaptive(
        peak_freqs, peak_amps, widths_hz, peaks_idx, freq_resolution
    )
    
    # 필터링된 결과 추출
    filtered_candidates.sort(key=lambda x: x['freq'])
    identified_modes_hz = np.array([c['freq'] for c in filtered_candidates])
    identified_indices = np.array([c['idx'] for c in filtered_candidates])
    identified_prominences = [] # 최종 선택된 피크의 돌출부 재추출
    
    # 플로팅을 위해 올바른 돌출부 값을 다시 매핑
    original_idx_list = list(peaks_idx)
    for c in filtered_candidates:
        # 이 인덱스가 원본 배열 어디에 있었는지 찾기
        if c['idx'] in original_idx_list:
            pos = original_idx_list.index(c['idx'])
            identified_prominences.append(prominences[pos])
        else:
            identified_prominences.append(0) # 발생해서는 안 됨
    identified_prominences = np.array(identified_prominences)

    # 항상 3개 모드로 고정, 부족하면 0 채우기
    TARGET_N_MODES = 3
    if len(identified_modes_hz) < TARGET_N_MODES:
        n_missing = TARGET_N_MODES - len(identified_modes_hz)
        identified_modes_hz = np.concatenate([identified_modes_hz, np.zeros(n_missing)])
        # filtered_candidates도 더미 추가 (search_bins 생성용)
        for _ in range(n_missing):
            filtered_candidates.append({'freq': 0.0, 'amp': 0.0, 'width': 0.0, 'idx': -1})

    logging.info(f"[{sensor_id}] 최종 고유 모드 (Hz): {np.round(identified_modes_hz, 3)}")

    # 4. 반-돌출부 너비를 사용하여 탐색 빈 정의
    search_bins = []
    for candidate in filtered_candidates:
        center_freq = candidate['freq']
        if center_freq == 0.0:  # 더미 모드는 skip
            search_bins.append((0.0, 0.0))  # 플레이스홀더
            continue
        half_width = candidate['width'] / 2.0  # width는 양쪽 합이므로 절반씩
        bin_min = max(PLOT_FREQ_MIN, center_freq - half_width)
        bin_max = min(PLOT_FREQ_MAX, center_freq + half_width)
        search_bins.append((bin_min, bin_max))

    logging.info(f"[{sensor_id}] 탐색 빈 (반-돌출부 너비): {[(round(a,2), round(b,2)) for a,b in search_bins]}")
    mode_column_names = ['1st_mode', '2nd_mode', '3rd_mode']

    # --- PASS 2 --- (개별 FFT에서 모드 주파수 추출)
    hourly_freq_data_for_excel = []
    for hour, df, xf_orig, power_orig in tqdm(hourly_fft_data_cache, desc=f"Pass 2/2 [{sensor_id}]"):
        if df is None: 
            empty_data = {'time': hour}
            for col in mode_column_names: empty_data[col] = 0.0  # 보수적 접근: NaN 대신 0
            hourly_freq_data_for_excel.append(empty_data)
            continue
        # find_hourly_peaks_in_bins에서 FeatureExtractor는 사용하지 않으므로 (xf_orig, power_orig가 이미 계산됨)
        # extractor 객체를 더미로 전달하거나 (find_hourly_peaks_in_bins 수정 없이) df를 전달하여 FFT를 재계산하도록 함
        # 현재 구현은 df를 사용하여 FFT를 재계산하므로 extractor를 전달
        hourly_peaks = find_hourly_peaks_in_bins(hour, df, extractor, search_bins, mode_column_names)
        hourly_freq_data_for_excel.append(hourly_peaks)

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

    freq_trend_df = pd.DataFrame(hourly_freq_data_for_excel)
    if not freq_trend_df.empty:
        freq_trend_df = freq_trend_df.sort_values(by='time')
        excel_filename = sensor_output_dir / f"{sensor_id}_frequency_trend_adaptive.xlsx"
        freq_trend_df.to_excel(excel_filename, index=False, sheet_name='Hourly_Modes')

    try:
        fig1, ax1 = plt.subplots(figsize=(16, 8))
        plot_average_fft(ax1, xf_master, mean_power, p05_power, p95_power, 
                          identified_indices, identified_prominences,
                          title_info, show_peaks=True)
        fig1.suptitle(f"Sensor Analysis (Adaptive): {sensor_id}", fontsize=16, fontweight='bold')
        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig1.savefig(sensor_output_dir / f"{sensor_id}_1_avg_fft_adaptive.png", dpi=150)
        plt.close(fig1)
    except Exception as e:
        logging.error(f"Plot 1 Error: {e}")

    try:
        fig3, ax3 = plt.subplots(figsize=(16, 8))
        plot_frequency_trend(ax3, freq_trend_df, identified_modes_hz, title_info) 
        fig3.suptitle(f"Frequency Trend (Adaptive Separation): {sensor_id}", fontsize=16, fontweight='bold')
        fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig3.savefig(sensor_output_dir / f"{sensor_id}_3_freq_trend_adaptive.png", dpi=150)
        plt.close(fig3)
    except Exception as e:
        logging.error(f"Plot 3 Error: {e}")

def process_sensor_wrapper(args):
    """
    단일 센서 폴더 처리를 위한 래퍼 함수 (병렬 처리에 사용)
    센서 유형을 분류하고 해당 처리 함수를 호출
    """
    sensor_dir, base_output_dir = args
    sensor_id = sensor_dir.name
    sensor_type = classify_sensor_type(sensor_id)
    
    try:
        if sensor_type == 'acceleration':
            # 가속도 센서
            logging.info(f"[{sensor_id}] ACCELERATION 센서로 감지됨")
            process_sensor_folder(sensor_dir, base_output_dir)
            
        elif sensor_type == 'static':
            # 정적 센서
            logging.info(f"[{sensor_id}] STATIC 센서로 감지됨")
            process_static_sensor_folder(sensor_dir, base_output_dir)
            
        else:  # sensor_type == 'unknown'
            # 알 수 없는 경우: 가속도로 시도 후 실패하면 정적으로 폴백
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
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    sensor_dirs = [d for d in BASE_DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not sensor_dirs:
        logging.error("BASE_DATA_DIR에서 센서 디렉토리를 찾을 수 없습니다.")
        return
    
    sensor_args = [(d, BASE_OUTPUT_DIR) for d in sensor_dirs]
    # 센서 레벨에서 병렬 처리
    with Pool(N_WORKERS_SENSORS) as pool:
        list(tqdm(pool.imap(process_sensor_wrapper, sensor_args), total=len(sensor_dirs), desc="센서 처리 중"))

if __name__ == "__main__":
    main()