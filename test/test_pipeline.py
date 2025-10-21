# tests/test_pipeline.py
"""
pipeline.py 통합 테스트
"""

import sys
from pathlib import Path
import yaml
import tempfile

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import ProcessingPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_config() -> dict:
    """테스트용 설정 생성"""
    return {
        'data': {
            'root_dir': 'Z:/05. Data/01. Under_Process/027. KICT_BMAPS/upload',
            'mode': 'stream',
            'file_pattern': '*.zip'
        },
        'features': {
            'acceleration': {
                'extract': ['ptp', 'noise', 'fft'],
                'sampling_rate': 100,
                'fft': {
                    'freq_range': [0.5, 20],
                    'n_peaks': 3,
                    'prominence_ratio': 0.1
                },
                'noise': {
                    'method': 'static_std',
                    'static_threshold': 0.01
                }
            }
        },
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'bridge_monitoring_test',
            'user': 'postgres',
            'password': 'postgres',
            'schema': 'public',
            'on_conflict': 'skip'
        },
        'processing': {
            'batch_size': 10,
            'max_files': 5,  # 테스트용
            'n_workers': 2,
            'reprocess': False,
            'skip_existing': False
        },
        'monitoring': {
            'enable': True,
            'report_interval': 1
        }
    }


def test_pipeline_initialization():
    """파이프라인 초기화 테스트"""
    print("\n=== Test 1: Pipeline Initialization ===")
    
    try:
        config = create_test_config()
        pipeline = ProcessingPipeline(config)
        
        print("✓ Pipeline initialized")
        print(f"  Zip handler mode: {pipeline.zip_handler.mode}")
        print(f"  Feature extractor sampling rate: {pipeline.feature_extractor.sampling_rate}")
        print(f"  Database: {pipeline.db.database}")
        
        pipeline._cleanup()
        return True
    
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_run():
    """파이프라인 실행 테스트"""
    print("\n=== Test 2: Pipeline Run (Small Scale) ===")
    
    try:
        config = create_test_config()
        config['processing']['max_files'] = 3  # 3개만 테스트
        
        pipeline = ProcessingPipeline(config)
        results = pipeline.run()
        
        print("✓ Pipeline completed")
        print(f"  Files processed: {results['files']['processed']}")
        print(f"  Records saved: {results['records']['saved']}")
        print(f"  Duration: {results['duration_formatted']}")
        print(f"  Performance: {results['performance']['files_per_second']:.2f} files/sec")
        
        if results['errors']:
            print(f"  Errors: {results['errors']}")
        
        return True
    
    except Exception as e:
        print(f"✗ Pipeline run failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint():
    """체크포인트 테스트"""
    print("\n=== Test 3: Checkpoint Save/Load ===")
    
    try:
        config = create_test_config()
        pipeline = ProcessingPipeline(config)
        
        # 초기 상태 저장
        pipeline._setup()
        pipeline.state.processed_files = 10
        pipeline.state.total_files = 100
        pipeline._save_checkpoint()
        
        print("✓ Checkpoint saved")
        
        # 새 파이프라인으로 로드
        pipeline2 = ProcessingPipeline(config)
        resumed = pipeline2._check_restart()
        
        if resumed:
            print("✓ Checkpoint loaded")
            print(f"  Resumed at: {pipeline2.state.processed_files} files")
        else:
            print("✗ Failed to load checkpoint")
            return False
        
        pipeline._cleanup()
        pipeline2._cleanup()
        
        return True
    
    except Exception as e:
        print(f"✗ Checkpoint test failed: {e}")
        return False


def test_error_handling():
    """에러 핸들링 테스트"""
    print("\n=== Test 4: Error Handling ===")
    
    try:
        config = create_test_config()
        # 존재하지 않는 디렉토리 설정
        config['data']['root_dir'] = '/nonexistent/path'
        
        pipeline = ProcessingPipeline(config)
        
        try:
            results = pipeline.run()
            print("✗ Should have raised an error")
            return False
        except FileNotFoundError:
            print("✓ Error correctly handled (FileNotFoundError)")
            return True
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_database_integration():
    """DB 통합 테스트"""
    print("\n=== Test 5: Database Integration ===")
    
    try:
        config = create_test_config()
        config['processing']['max_files'] = 2
        
        pipeline = ProcessingPipeline(config)
        
        # DB 초기화
        pipeline.db.create_tables(drop_existing=False)
        print("✓ Database tables ready")
        
        # 파이프라인 실행
        results = pipeline.run()
        
        # DB 통계 확인
        db_stats = pipeline.get_database_statistics()
        print(f"✓ Database statistics retrieved")
        print(f"  Total sensors: {db_stats['total_sensors']}")
        
        if db_stats['sensors']:
            print(f"  Sample sensor: {db_stats['sensors'][0]['sensor_id']}")
            print(f"    Records: {db_stats['sensors'][0]['n_records']}")
        
        return True
    
    except Exception as e:
        print(f"✗ Database integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("Pipeline Integration Tests")
    print("="*60)
    print("\nNote: These tests require:")
    print("  1. PostgreSQL/TimescaleDB running")
    print("  2. Access to data directory")
    print("  3. Test database: bridge_monitoring_test")
    print("="*60)
    
    results = []
    
    # 실행할 테스트
    results.append(("Initialization", test_pipeline_initialization()))
    results.append(("Checkpoint", test_checkpoint()))
    results.append(("Error Handling", test_error_handling()))
    
    # 실제 데이터가 있는 경우만
    # results.append(("Pipeline Run", test_pipeline_run()))
    # results.append(("Database Integration", test_database_integration()))
    
    # 결과 요약
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:25s} : {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)