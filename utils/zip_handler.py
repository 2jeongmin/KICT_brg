# src/zip_handler.py
"""
Zip 압축 파일 처리 유틸리티
용량 문제 없이 zip 파일을 스트리밍 방식으로 처리

주요 기능:
- 메모리 스트리밍: 디스크 압축 해제 없이 메모리에서 직접 처리
- 배치 처리: 여러 zip 파일을 효율적으로 처리
- 에러 핸들링: 손상된 파일 자동 건너뛰기
"""
import logging
from pathlib import Path
from typing import List
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Tuple
import zipfile
import io
import shutil
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame
import logging
from tqdm import tqdm

from .io_utils import parse_bin_filename

logger = logging.getLogger(__name__)


class ZipHandler:
    """
    Zip 압축 파일 처리 클래스
    
    3가지 처리 모드:
    - 'stream': 메모리에서 직접 읽기 (권장, 디스크 용량 증가 0)
    - 'extract_delete': 압축 해제 후 처리하고 원본 삭제
    - 'temp_dir': 임시 디렉토리에 압축 해제 후 처리
    """
    
    def __init__(self, 
                 mode: str = 'stream',
                 temp_dir: Optional[Path] = None,
                 keep_extracted: bool = False):
        self.logger = logging.getLogger(__name__) 
        """
        Args:
            mode: 처리 모드 ('stream', 'extract_delete', 'temp_dir')
            temp_dir: 임시 디렉토리 (temp_dir 모드에서만 사용)
            keep_extracted: 압축 해제된 파일 유지 여부
        """
        if mode not in ['stream', 'extract_delete', 'temp_dir']:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: stream, extract_delete, temp_dir")
        
        self.mode = mode
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self.keep_extracted = keep_extracted
        
        # temp_dir 모드이지만 경로가 지정되지 않은 경우
        if mode == 'temp_dir' and not self.temp_dir:
            self.temp_dir = Path(tempfile.gettempdir()) / "bridge_monitoring_temp"
            self.temp_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Using temporary directory: {self.temp_dir}")
        
        # 통계
        self.stats = {
            'total_zips': 0,
            'processed_zips': 0,
            'failed_zips': 0,
            'total_bins': 0,
            'processed_bins': 0,
            'failed_bins': 0
        }
    
    def find_zip_files(self, 
                       root_dir: Path, 
                       pattern: str = "*.zip",
                       recursive: bool = True) -> List[Path]:
        """
        디렉토리에서 zip 파일 찾기
        
        Args:
            root_dir: 검색 시작 디렉토리
            pattern: 파일 패턴
            recursive: 하위 디렉토리 포함 여부
        
        Returns:
            zip 파일 경로 리스트 (정렬됨)
        """
        root_path = Path(root_dir)
        
        if not root_path.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")
        
        # 디버깅: 디렉토리 구조 확인
        self.logger.info(f"Scanning directory: {root_path}")
        self.logger.info(f"Directory exists: {root_path.exists()}")
        self.logger.info(f"Is directory: {root_path.is_dir()}")
        
        # 하위 디렉토리 확인
        subdirs = [d for d in root_path.iterdir() if d.is_dir()]
        self.logger.info(f"Found {len(subdirs)} subdirectories")
        if subdirs:
            self.logger.info(f"First few subdirs: {[d.name for d in subdirs[:5]]}")

        zip_files = []
        
        if recursive:
            import os
            for dirpath, dirnames, filenames in os.walk(root_path):
                for filename in filenames:
                    if filename.endswith('.zip'):
                        zip_files.append(Path(dirpath) / filename)
        else:
            zip_files = list(root_path.glob(pattern))
        
        zip_files.sort()
        
        self.logger.info(f"Found {len(zip_files)} zip files")
        if zip_files:
            self.logger.info(f"First zip: {zip_files[0]}")
        
        return zip_files
    
    def get_bin_list_from_zip(self, zip_path: Path) -> List[str]:
        """
        zip 파일 내부의 bin 파일 목록 반환
        
        Args:
            zip_path: zip 파일 경로
        
        Returns:
            bin 파일명 리스트
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                bin_files = [f for f in zf.namelist() if f.endswith('.bin')]
                return bin_files
        except Exception as e:
            logger.error(f"Failed to read zip file {zip_path.name}: {e}")
            return []
    
    def parse_bin_from_bytes(self, bin_data: bytes) -> DataFrame:
        """
        Binary 데이터(bytes)를 DataFrame으로 변환
        
        Args:
            bin_data: bin 파일의 raw bytes
        
        Returns:
            DataFrame (columns=['time', 'value'])
        """
        # 파일 크기로 레코드 개수 계산
        file_size = len(bin_data)
        record_size = 12  # int64 (8bytes) + float32 (4bytes)
        n_records = file_size // record_size
        
        if n_records == 0:
            logger.warning("Empty bin data")
            return DataFrame(columns=['time', 'value'])
        
        # 배열 미리 할당
        timestamps = np.empty(n_records, dtype='int64')
        values = np.empty(n_records, dtype='float32')
        
        # numpy frombuffer로 파싱 (빠름!)
        for i in range(n_records):
            offset = i * record_size
            
            # Big-endian int64 (timestamp in milliseconds)
            timestamps[i] = np.frombuffer(
                bin_data[offset:offset+8], 
                dtype='>i8'
            )[0]
            
            # Big-endian float32 (acceleration value)
            values[i] = np.frombuffer(
                bin_data[offset+8:offset+12], 
                dtype='>f4'
            )[0]
        
        # datetime 변환
        times = pd.to_datetime(timestamps, unit='ms')
        
        # DataFrame 생성
        df = DataFrame({
            'time': times,
            'value': values
        })
        
        return df
    
    def process_zip_file_stream(self, zip_path: Path) -> List[Dict]:
        """
        [스트리밍 모드] zip 파일 내 모든 bin 파일을 메모리에서 처리
        
        디스크에 압축 해제하지 않음 → 용량 증가 0
        
        Args:
            zip_path: zip 파일 경로
        
        Returns:
            각 bin 파일의 처리 결과 리스트
            [
                {
                    'zip_file': str,
                    'bin_file': str,
                    'sensor_id': str,
                    'hour': datetime,
                    'n_records': int,
                    'df': DataFrame,
                    'file_size': int,
                    'metadata': dict
                },
                ...
            ]
        """
        results = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # zip 내 bin 파일 목록
                bin_files = [f for f in zf.namelist() if f.endswith('.bin')]
                
                if not bin_files:
                    logger.warning(f"No bin files found in {zip_path.name}")
                    self.stats['failed_zips'] += 1
                    return results
                
                logger.debug(f"Processing {len(bin_files)} bin files from {zip_path.name}")
                
                # 각 bin 파일 처리
                for bin_filename in bin_files:
                    try:
                        # 메모리로 읽기
                        with zf.open(bin_filename) as bin_file:
                            bin_data = bin_file.read()
                        
                        # 파싱
                        df = self.parse_bin_from_bytes(bin_data)
                        
                        # 파일명 파싱
                        bin_path = Path(bin_filename)
                        metadata = parse_bin_filename(bin_path)
                        
                        result = {
                            'zip_file': zip_path.name,
                            'bin_file': bin_filename,
                            'sensor_id': metadata['sensor_id'],
                            'hour': metadata['hour'],
                            'n_records': len(df),
                            'df': df,
                            'file_size': len(bin_data),
                            'metadata': metadata
                        }
                        
                        results.append(result)
                        self.stats['processed_bins'] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process {bin_filename} from {zip_path.name}: {e}")
                        self.stats['failed_bins'] += 1
                        continue
                
                self.stats['processed_zips'] += 1
        
        except zipfile.BadZipFile as e:
            logger.error(f"Bad zip file {zip_path.name}: {e}")
            self.stats['failed_zips'] += 1
        
        except Exception as e:
            logger.error(f"Failed to open {zip_path.name}: {e}")
            self.stats['failed_zips'] += 1
        
        return results
    
    def extract_and_process(self, 
                           zip_path: Path,
                           delete_after: bool = False) -> List[Dict]:
        """
        [압축 해제 모드] zip 파일 압축 해제 후 처리
        
        Args:
            zip_path: zip 파일 경로
            delete_after: 처리 후 압축 해제된 파일 삭제 여부
        
        Returns:
            각 bin 파일의 처리 결과
        """
        # 압축 해제 디렉토리 결정
        if self.mode == 'temp_dir':
            extract_dir = self.temp_dir / zip_path.stem
        else:
            # 같은 디렉토리에 압축 해제
            extract_dir = zip_path.parent / zip_path.stem
        
        extract_dir.mkdir(exist_ok=True, parents=True)
        results = []
        
        try:
            # 압축 해제
            logger.debug(f"Extracting {zip_path.name} to {extract_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            # bin 파일 찾기
            bin_files = list(extract_dir.glob("*.bin"))
            
            if not bin_files:
                logger.warning(f"No bin files extracted from {zip_path.name}")
                self.stats['failed_zips'] += 1
                return results
            
            # 각 bin 파일 처리
            for bin_path in bin_files:
                try:
                    # 파일 읽기
                    with open(bin_path, 'rb') as f:
                        bin_data = f.read()
                    
                    # 파싱
                    df = self.parse_bin_from_bytes(bin_data)
                    metadata = parse_bin_filename(bin_path)
                    
                    result = {
                        'zip_file': zip_path.name,
                        'bin_file': bin_path.name,
                        'sensor_id': metadata['sensor_id'],
                        'hour': metadata['hour'],
                        'n_records': len(df),
                        'df': df,
                        'file_size': len(bin_data),
                        'metadata': metadata
                    }
                    
                    results.append(result)
                    self.stats['processed_bins'] += 1
                
                except Exception as e:
                    logger.error(f"Failed to process {bin_path.name}: {e}")
                    self.stats['failed_bins'] += 1
                    continue
            
            self.stats['processed_zips'] += 1
        
        except Exception as e:
            logger.error(f"Failed to extract {zip_path.name}: {e}")
            self.stats['failed_zips'] += 1
        
        finally:
            # 정리
            if delete_after or (self.mode == 'temp_dir' and not self.keep_extracted):
                try:
                    shutil.rmtree(extract_dir)
                    logger.debug(f"Cleaned up {extract_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {extract_dir}: {e}")
            
            # extract_delete 모드면 원본 zip도 삭제
            if self.mode == 'extract_delete' and delete_after:
                try:
                    zip_path.unlink()
                    logger.info(f"Deleted original zip: {zip_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {zip_path.name}: {e}")
        
        return results
    
    def process_single_zip(self, zip_path: Path) -> List[Dict]:
        """
        단일 zip 파일 처리 (모드에 따라 자동 선택)
        
        Args:
            zip_path: zip 파일 경로
        
        Returns:
            처리 결과 리스트
        """
        self.stats['total_zips'] += 1
        
        if self.mode == 'stream':
            return self.process_zip_file_stream(zip_path)
        else:
            delete = (self.mode == 'extract_delete')
            return self.extract_and_process(zip_path, delete_after=delete)
    
    def process_all_zips(self, 
                        root_dir: Path,
                        pattern: str = "*.zip",
                        max_files: Optional[int] = None,
                        show_progress: bool = True) -> Iterator[List[Dict]]:
        """
        디렉토리의 모든 zip 파일 처리 (generator)
        
        Args:
            root_dir: 루트 디렉토리
            pattern: zip 파일 패턴
            max_files: 처리할 최대 파일 수 (테스트용)
            show_progress: 진행률 표시 여부
        
        Yields:
            각 zip 파일의 처리 결과 리스트
        """
        # zip 파일 목록
        zip_files = self.find_zip_files(root_dir, pattern)
        
        if max_files:
            zip_files = zip_files[:max_files]
            logger.info(f"Limited to {max_files} files (test mode)")
        
        self.stats['total_zips'] = len(zip_files)
        
        logger.info(f"Processing {len(zip_files)} zip files in '{self.mode}' mode")
        
        # 진행률 표시
        iterator = tqdm(zip_files, desc="Processing zip files") if show_progress else zip_files
        
        for zip_path in iterator:
            try:
                results = self.process_single_zip(zip_path)
                
                if results:
                    self.stats['total_bins'] += len(results)
                    yield results
                    
            except KeyboardInterrupt:
                logger.warning("Processing interrupted by user")
                raise
            
            except Exception as e:
                logger.error(f"Fatal error processing {zip_path.name}: {e}")
                continue
    
    def estimate_disk_usage(self, root_dir: Path) -> Dict[str, float]:
        """
        디스크 사용량 추정
        
        Args:
            root_dir: 루트 디렉토리
        
        Returns:
            용량 정보 딕셔너리 (GB 단위)
        """
        zip_files = self.find_zip_files(root_dir)
        
        total_zip_size = sum(f.stat().st_size for f in zip_files)
        
        # 압축률 가정: bin 파일은 약 30-40%로 압축됨
        compression_ratio = 3.0
        estimated_uncompressed = total_zip_size * compression_ratio
        
        info = {
            'total_zip_gb': total_zip_size / (1024**3),
            'estimated_uncompressed_gb': estimated_uncompressed / (1024**3),
            'n_files': len(zip_files),
            'avg_file_size_mb': (total_zip_size / len(zip_files) / (1024**2)) if zip_files else 0
        }
        
        return info
    
    def get_statistics(self) -> Dict:
        """
        처리 통계 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        stats = self.stats.copy()
        
        # 성공률 계산
        if stats['total_zips'] > 0:
            stats['zip_success_rate'] = stats['processed_zips'] / stats['total_zips']
        else:
            stats['zip_success_rate'] = 0.0
        
        if stats['total_bins'] > 0:
            stats['bin_success_rate'] = stats['processed_bins'] / stats['total_bins']
        else:
            stats['bin_success_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """통계 초기화"""
        for key in self.stats:
            self.stats[key] = 0


# 편의 함수들

def quick_process_zips(root_dir: Path, 
                       mode: str = 'stream',
                       max_files: Optional[int] = None,
                       temp_dir: Optional[Path] = None) -> List[Dict]:
    """
    간단한 zip 파일 처리 래퍼 함수
    
    Args:
        root_dir: 루트 디렉토리
        mode: 처리 모드
        max_files: 최대 파일 수
        temp_dir: 임시 디렉토리
    
    Returns:
        모든 처리 결과 리스트
    
    Example:
        >>> results = quick_process_zips(Path("Z:/data"), mode='stream', max_files=10)
        >>> print(f"Processed {len(results)} bin files")
    """
    handler = ZipHandler(mode=mode, temp_dir=temp_dir)
    
    # 디스크 사용량 확인
    usage = handler.estimate_disk_usage(root_dir)
    logger.info(f"Disk usage estimate:")
    logger.info(f"  Compressed: {usage['total_zip_gb']:.2f} GB")
    logger.info(f"  Uncompressed: {usage['estimated_uncompressed_gb']:.2f} GB")
    logger.info(f"  Files: {usage['n_files']}")
    
    # 처리
    all_results = []
    for results in handler.process_all_zips(root_dir, max_files=max_files):
        all_results.extend(results)
    
    # 통계
    stats = handler.get_statistics()
    logger.info(f"Processing complete:")
    logger.info(f"  Zip files: {stats['processed_zips']}/{stats['total_zips']}")
    logger.info(f"  Bin files: {stats['processed_bins']}/{stats['total_bins']}")
    logger.info(f"  Success rate: {stats['zip_success_rate']*100:.1f}%")
    
    return all_results


def get_zip_info(zip_path: Path) -> Dict:
    """
    zip 파일 정보 조회
    
    Args:
        zip_path: zip 파일 경로
    
    Returns:
        파일 정보 딕셔너리
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            bin_files = [f for f in zf.namelist() if f.endswith('.bin')]
            
            info = {
                'zip_file': zip_path.name,
                'zip_size': zip_path.stat().st_size,
                'n_bin_files': len(bin_files),
                'bin_files': bin_files
            }
            
            # 압축률 계산
            if bin_files:
                total_uncompressed = sum(zf.getinfo(f).file_size for f in bin_files)
                info['total_uncompressed_size'] = total_uncompressed
                info['compression_ratio'] = total_uncompressed / info['zip_size']
            
            return info
    
    except Exception as e:
        logger.error(f"Failed to get info for {zip_path.name}: {e}")
        return {}


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python zip_handler.py <root_dir> [mode] [max_files]")
        print("\nModes:")
        print("  stream (default) - 메모리 스트리밍, 디스크 용량 증가 0")
        print("  extract_delete   - 압축 해제 후 처리, 원본 삭제")
        print("  temp_dir         - 임시 디렉토리에 압축 해제")
        print("\nExample:")
        print("  python zip_handler.py Z:/data stream 10")
        sys.exit(1)
    
    # 인자 파싱
    root = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else 'stream'
    max_files = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    print(f"\n{'='*60}")
    print(f"Zip Handler Test")
    print(f"{'='*60}")
    print(f"Directory: {root}")
    print(f"Mode: {mode}")
    if max_files:
        print(f"Max files: {max_files}")
    print(f"{'='*60}\n")
    
    # 처리
    results = quick_process_zips(root, mode=mode, max_files=max_files)
    
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total bin files processed: {len(results)}")
    
    # 센서별 통계
    if results:
        sensor_counts = {}
        for r in results:
            sensor_id = r['sensor_id']
            sensor_counts[sensor_id] = sensor_counts.get(sensor_id, 0) + 1
        
        print(f"\nFiles per sensor:")
        for sensor, count in sorted(sensor_counts.items())[:10]:  # 상위 10개만
            print(f"  {sensor}: {count} files")
        
        if len(sensor_counts) > 10:
            print(f"  ... and {len(sensor_counts) - 10} more sensors")