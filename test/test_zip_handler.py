# tests/test_zip_handler.py
"""
zip_handler.py 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.zip_handler import ZipHandler, get_zip_info, quick_process_zips
import logging

logging.basicConfig(level=logging.INFO)


def test_single_zip():
    """단일 zip 파일 테스트"""
    print("\n=== Test 1: Single Zip File ===")
    
    # 테스트할 zip 파일 경로 (실제 경로로 변경)
    zip_file = Path("Z:/05. Data/01. Under_Process/027. KICT_BMAPS/upload/smartcs_2_DNAGW2109_202405211700.zip")
    
    if not zip_file.exists():
        print(f"Skip: File not found - {zip_file}")
        return
    
    # zip 정보 확인
    info = get_zip_info(zip_file)
    print(f"Zip info: {info}")
    
    # 스트리밍 처리
    handler = ZipHandler(mode='stream')
    results = handler.process_single_zip(zip_file)
    
    print(f"Processed {len(results)} bin files")
    
    for result in results:
        print(f"  {result['sensor_id']} - {result['hour']} - {result['n_records']} records")
        print(f"    DataFrame shape: {result['df'].shape}")
        print(f"    Sample data:\n{result['df'].head()}")


def test_batch_processing():
    """배치 처리 테스트"""
    print("\n=== Test 2: Batch Processing ===")
    
    root_dir = Path("Z:/05. Data/01. Under_Process/027. KICT_BMAPS/upload")
    
    if not root_dir.exists():
        print(f"Skip: Directory not found - {root_dir}")
        return
    
    # 10개 파일만 테스트
    handler = ZipHandler(mode='stream')
    
    all_results = []
    for results in handler.process_all_zips(root_dir, max_files=10):
        all_results.extend(results)
        print(f"Batch processed: {len(results)} bin files")
    
    # 통계
    stats = handler.get_statistics()
    print(f"\nStatistics:")
    print(f"  Zip files: {stats['processed_zips']}/{stats['total_zips']}")
    print(f"  Bin files: {stats['processed_bins']}/{stats['total_bins']}")
    print(f"  Success rate: {stats['zip_success_rate']*100:.1f}%")


def test_disk_usage():
    """디스크 사용량 추정 테스트"""
    print("\n=== Test 3: Disk Usage Estimation ===")
    
    root_dir = Path("Z:/05. Data/01. Under_Process/027. KICT_BMAPS/upload")
    
    if not root_dir.exists():
        print(f"Skip: Directory not found - {root_dir}")
        return
    
    handler = ZipHandler(mode='stream')
    usage = handler.estimate_disk_usage(root_dir)
    
    print(f"Compressed: {usage['total_zip_gb']:.2f} GB")
    print(f"Estimated uncompressed: {usage['estimated_uncompressed_gb']:.2f} GB")
    print(f"Number of files: {usage['n_files']}")
    print(f"Average file size: {usage['avg_file_size_mb']:.2f} MB")


if __name__ == "__main__":
    print("="*60)
    print("Zip Handler Tests")
    print("="*60)
    
    # 실행할 테스트 선택
    test_single_zip()
    test_batch_processing()
    test_disk_usage()
    
    print("\n" + "="*60)
    print("All tests completed")
    print("="*60)