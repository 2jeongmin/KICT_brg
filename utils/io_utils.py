"""
Binary 파일 I/O 유틸리티
교량 센서 데이터 bin 파일을 읽고 처리
"""
import struct
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame
import logging

logger = logging.getLogger(__name__)

def get_df_from_bin(filepath: Path, validate: bool = True) -> DataFrame:
    """
    Binary 파일을 DataFrame으로 변환

    파일 형식: [int64(timestamp_ms, big-endian) + float32(value, big-endian)] 반복
    
    Args:
        filepath: bin 파일 경로
        validate: 데이터 검증 수행 여부
    
    Returns:
        columns=['time', 'value']인 DataFrame
        - time: datetime64[ns]
        - value: float32
    
    Raises:
        FileNotFoundError: 파일이 없는 경우
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # 파일 크기로 데이터 개수 계산 (8 bytes + 4 bytes = 12 bytes per record)
    file_size = filepath.stat().st_size
    record_size = 12  # int64 + float32
    
    if file_size % record_size != 0:
        logger.warning(f"File size not aligned to record size: {filepath.name}")
    
    n_records = file_size // record_size
    
    if n_records == 0:
        logger.warning(f"Empty file: {filepath.name}")
        return DataFrame(columns=['time', 'value'])
    
    # numpy 배열 미리 할당
    timestamps = np.empty(n_records, dtype='int64')
    values = np.empty(n_records, dtype='float32')
    
    try:
        with open(filepath, 'rb') as f:
            # 전체 파일을 한번에 읽기
            data = f.read()
            
            # numpy.frombuffer로 한번에 파싱
            # big-endian이므로 '>i8' (int64), '>f4' (float32)
            for i in range(n_records):
                offset = i * record_size
                timestamps[i] = np.frombuffer(
                    data[offset:offset+8], 
                    dtype='>i8'
                )[0]
                values[i] = np.frombuffer(
                    data[offset+8:offset+12], 
                    dtype='>f4'
                )[0]
        
        # timestamp를 datetime으로 변환 (milliseconds → datetime64[ns])
        times = pd.to_datetime(timestamps, unit='ms')
        
        # DataFrame 생성
        df = DataFrame({
            'time': times,
            'value': values
        })
        
        # 데이터 검증
        if validate:
            issues = _validate_dataframe(df, filepath)
            if issues:
                logger.warning(f"Validation issues in {filepath.name}: {issues}")
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to read {filepath.name}: {e}")
        raise


def _validate_dataframe(df: DataFrame, filepath: Path) -> Dict[str, bool]:
    """
    DataFrame 데이터 품질 검증
    """
    issues = {}
    
    if len(df) == 0:
        issues['empty'] = True
        return issues
    
    # 1. 데이터 개수 체크 (시간당 360,000개 기준, ±10% 허용)
    if len(df) > 400000:
        issues['too_many_records'] = True
    elif len(df) < 320000:
        issues['too_few_records'] = True
    
    # 2. 시간 간격 체크 (100Hz → ~10ms)
    if len(df) >= 2:
        time_diff = (df['time'].iloc[1] - df['time'].iloc[0]).total_seconds() * 1000
        if time_diff == 0:
            issues['zero_time_delta'] = True
        elif time_diff < 9 or time_diff > 11:
            issues['abnormal_time_delta'] = True
    
    # 3. 값 범위 체크 (가속도: 일반적으로 -100 ~ 100 m/s^2)
    values = df['value'].values
    if not np.all(np.isfinite(values)):
        issues['non_finite_values'] = True
    
    min_val, max_val = values.min(), values.max()
    if min_val < -100 or max_val > 100:
        issues['out_of_range'] = True
    
    # 4. 시간 정렬 체크
    if not df['time'].is_monotonic_increasing:
        issues['time_not_sorted'] = True
    
    return issues


def find_bin_files(root_dir: Path, 
                   pattern: str = "*.bin",
                   recursive: bool = True) -> List[Path]:
    """
    디렉토리에서 bin 파일 찾기
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    if recursive:
        files = list(root_path.rglob(pattern))
    else:
        files = list(root_path.glob(pattern))
    
    # 파일명으로 정렬
    files.sort()
    
    logger.info(f"Found {len(files)} bin files in {root_dir}")
    return files


# ===============================================
# CRITICAL FIX: 모든 파일명 예외 처리 지원
# ===============================================
def parse_bin_filename(filepath: Path) -> Dict[str, any]:
    """
    bin 파일명에서 메타데이터 추출 (모든 예외 상황 처리 버전)
    
    지원 패턴:
    1. Standard (10자리 시간): bmaps_1_DNA22026_2024102901.zip
    2. Standard (12자리 시간): bmaps_1_gcskyb01_202410251800.zip
    3. Extra Parts (gnss1):    bmaps_5_gnss1_0_2024102519.zip
    4. Corrupted/Extended:     bmaps_2_DNAGW22019_99_1_2025031704.bin
    
    Args:
        filepath: bin 파일 경로
    
    Returns:
        메타데이터 딕셔너리
    """
    filename = filepath.stem  # 확장자 제거
    parts = filename.split('_')
    
    # 최소 4개의 파트가 있어야 함 [system, id, sensor, ..., time]
    if len(parts) < 4:
        raise ValueError(f"Invalid filename format (less than 4 parts): {filepath.name}")
    
    try:
        # 1. 시간 문자열 추출 (항상 마지막 파트)
        time_str = parts[-1]
        
        # 시간 포맷 결정 (10자리 또는 12자리)
        if len(time_str) == 10:  # YYYYMMDDHH
            time_fmt = "%Y%m%d%H"
        elif len(time_str) == 12:  # YYYYMMDDHHMM
            time_fmt = "%Y%m%d%H%M"
        else:
            # 알 수 없는 시간 형식이지만, 숫자라면 일단 ValueError를 내고 로그 확인 유도
            raise ValueError(f"Unknown timestamp format (length {len(time_str)}): {time_str}")
            
        timestamp = datetime.strptime(time_str, time_fmt)
        
        # 2. 센서 ID 추출 (항상 3번째 파트, index 2)
        # 예: bmaps_5_gnss1_0_... -> 'gnss1'
        # 예: bmaps_1_DNA22026_... -> 'DNA22026'
        sensor_id = parts[2]
        
        # 3. 시스템 및 시스템 ID
        system = parts[0]
        try:
            system_id = int(parts[1])
        except ValueError:
            system_id = parts[1] # 숫자가 아니면 문자열 그대로
            
        return {
            'system': system,
            'system_id': system_id,
            'sensor_id': sensor_id,
            'timestamp': timestamp,
            'hour': timestamp.replace(minute=0, second=0, microsecond=0), # 분석용 시간(시 단위 절삭)
            'filename': filepath.name
        }
    
    except Exception as e:
        raise ValueError(f"Failed to parse filename '{filepath.name}': {e}")


def get_file_info_batch(filepaths: List[Path]) -> DataFrame:
    """
    여러 파일의 메타데이터를 배치로 추출
    """
    records = []
    
    for filepath in filepaths:
        try:
            info = parse_bin_filename(filepath)
            info['file_size'] = filepath.stat().st_size
            info['filepath'] = str(filepath)
            records.append(info)
        except Exception as e:
            logger.warning(f"Skipping {filepath.name}: {e}")
    
    df = DataFrame(records)
    return df


def estimate_processing_time(n_files: int, 
                            files_per_second: float = 3.0) -> str:
    """
    예상 처리 시간 계산
    """
    total_seconds = n_files / files_per_second
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    if hours > 0:
        return f"{hours}시간 {minutes}분"
    else:
        return f"{minutes}분"


if __name__ == "__main__":
    # 간단한 테스트
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        
        print(f"\n=== Testing {test_file.name} ===")
        
        try:
            # 파일명 파싱 테스트
            info = parse_bin_filename(test_file)
            print(f"\n파일 정보 (파일명 파싱 성공):")
            for key, value in info.items():
                print(f"  {key}: {value}")
            
            # 데이터 읽기 테스트
            import time
            start = time.time()
            df = get_df_from_bin(test_file)
            elapsed = time.time() - start
            
            print(f"\n데이터 읽기 완료:")
            print(f"  행 개수: {len(df):,}")
            print(f"  소요 시간: {elapsed:.3f}초")
            
        except Exception as e:
            print(f"\n!!! ERROR 발생: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print("Usage: python io_utils.py <bin_file_path>")