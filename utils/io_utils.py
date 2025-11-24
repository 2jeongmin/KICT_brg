"""
Binary 파일 I/O 유틸리티
교량 센서 데이터 bin 파일을 읽고 처리
"""
import struct
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame
import logging

logger = logging.getLogger(__name__)

def get_df_from_bin(filepath: Path) -> pd.DataFrame:
    """
    bin 파일을 DataFrame으로 변환
    
    Format: [int64(timestamp_ms, big-endian) + float32(value, big-endian)] 반복
    
    Returns:
        DataFrame with columns ['time', 'value']
    """
    file_size = filepath.stat().st_size
    record_size = 8 + 4  # int64 + float32
    n_records = file_size // record_size
    
    # 미리 할당
    timestamps = np.empty(n_records, dtype='int64')
    values = np.empty(n_records, dtype='float32')
    
    with open(filepath, 'rb') as f:
        for i in range(n_records):
            ts_bytes = f.read(8)
            val_bytes = f.read(4)
            
            if len(ts_bytes) < 8 or len(val_bytes) < 4:
                break
                
            timestamps[i] = struct.unpack('>q', ts_bytes)[0]
            values[i] = struct.unpack('>f', val_bytes)[0]
    
    # DataFrame 생성
    df = pd.DataFrame({
        'time': pd.to_datetime(timestamps[:i], unit='ms'),
        'value': values[:i]
    })
    
    return df

def parse_bin_filename(filepath: Path) -> dict:
    """
    bin 파일명 파싱
    
    Format: smartcs_1_DNA21001_2024102520.bin
    
    Returns:
        {'system': 'smartcs', 'id': '1', 'sensor': 'DNA21001', 
         'year': 2024, 'month': 10, 'day': 25, 'hour': 20}
    """
    parts = filepath.stem.split('_')
    
    if len(parts) < 4:
        raise ValueError(f"Invalid filename format: {filepath.name}")
    
    system = parts[0]
    id_num = parts[1]
    sensor = parts[2]
    datetime_str = parts[3]
    
    return {
        'system': system,
        'id': id_num,
        'sensor': sensor,
        'year': int(datetime_str[:4]),
        'month': int(datetime_str[4:6]),
        'day': int(datetime_str[6:8]),
        'hour': int(datetime_str[8:10]),
        'filename': filepath.name
    }

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
        ValueError: 파일 크기가 비정상인 경우
    
    Performance:
        - 시간당 파일(~360,000 rows): ~0.3초
        - 메모리: 파일당 약 3MB
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
    
    Returns:
        검증 결과 딕셔너리 (이상 있는 항목만 포함)
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
    
    Args:
        root_dir: 검색 시작 디렉토리
        pattern: 파일 패턴 (기본값: "*.bin")
        recursive: 하위 디렉토리 포함 여부
    
    Returns:
        bin 파일 경로 리스트 (정렬됨)
    
    Example:
        >>> files = find_bin_files(Path("data/raw"))
        >>> len(files)
        50000
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


def parse_bin_filename(filepath: Path) -> Dict[str, any]:
    """
    bin 파일명에서 메타데이터 추출
    
    파일명 패턴: {system}_{id}_{sensor}_{YYYYMMDDhh}.bin
    예시: smartcs_1_DNA21006_2024032620.bin
    
    Args:
        filepath: bin 파일 경로
    
    Returns:
        메타데이터 딕셔너리:
        - system: 시스템 ID (예: 'smartcs')
        - system_id: 시스템 번호 (예: 1)
        - sensor_id: 센서 ID (예: 'DNA21006')
        - timestamp: 시간 (datetime)
        - hour: 시간 (datetime, 분/초는 00:00)
        - filename: 원본 파일명
    
    Raises:
        ValueError: 파일명 형식이 맞지 않는 경우
    
    Example:
        >>> parse_bin_filename(Path("smartcs_1_DNA21006_2024032620.bin"))
        {
            'system': 'smartcs',
            'system_id': 1,
            'sensor_id': 'DNA21006',
            'timestamp': datetime(2024, 3, 26, 20, 0),
            'hour': datetime(2024, 3, 26, 20, 0),
            'filename': 'smartcs_1_DNA21006_2024032620.bin'
        }
    """
    filename = filepath.stem  # 확장자 제거
    parts = filename.split('_')
    
    if len(parts) < 4:
        raise ValueError(f"Invalid filename format: {filepath.name}")
    
    try:
        system = parts[0]
        system_id = int(parts[1])
        sensor_id = parts[2]
        time_str = parts[3]  # YYYYMMDDhh
        
        # 시간 파싱
        year = int(time_str[:4])
        month = int(time_str[4:6])
        day = int(time_str[6:8])
        hour = int(time_str[8:10])
        
        timestamp = datetime(year, month, day, hour)
        
        return {
            'system': system,
            'system_id': system_id,
            'sensor_id': sensor_id,
            'timestamp': timestamp,
            'hour': timestamp,  # 시간 단위
            'filename': filepath.name
        }
    
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse filename {filepath.name}: {e}")


def get_file_info_batch(filepaths: List[Path]) -> DataFrame:
    """
    여러 파일의 메타데이터를 배치로 추출
    
    Args:
        filepaths: bin 파일 경로 리스트
    
    Returns:
        파일 정보 DataFrame
        columns: ['filename', 'system', 'system_id', 'sensor_id', 'hour', 'file_size']
    
    Example:
        >>> files = find_bin_files(Path("data/raw"))
        >>> file_info = get_file_info_batch(files)
        >>> file_info.groupby('sensor_id').size()
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


# 편의 함수들
def estimate_processing_time(n_files: int, 
                            files_per_second: float = 3.0) -> str:
    """
    예상 처리 시간 계산
    
    Args:
        n_files: 처리할 파일 개수
        files_per_second: 초당 처리 파일 수
    
    Returns:
        예상 소요 시간
    
    Example:
        >>> estimate_processing_time(50000)
        '4시간 37분'
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
        
        # 파일명 파싱
        info = parse_bin_filename(test_file)
        print(f"\n파일 정보:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 데이터 읽기
        import time
        start = time.time()
        df = get_df_from_bin(test_file)
        elapsed = time.time() - start
        
        print(f"\n데이터 읽기 완료:")
        print(f"  행 개수: {len(df):,}")
        print(f"  소요 시간: {elapsed:.3f}초")
        print(f"  처리 속도: {len(df)/elapsed:,.0f} rows/sec")
        print(f"\n데이터 샘플:")
        print(df.head())
        print(f"\n기본 통계:")
        print(df.describe())
    else:
        print("Usage: python io_utils.py <bin_file_path>")