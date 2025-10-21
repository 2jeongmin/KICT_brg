# tests/test_database.py
"""
database.py 테스트
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database import TimeseriesDB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_connection():
    """DB 연결 테스트"""
    print("\n=== Test 1: Database Connection ===")
    
    try:
        db = TimeseriesDB(
            host='localhost',
            port=5432,
            database='bridge_monitoring_test',
            user='postgres',
            password='postgres'
        )
        print("✓ Connection successful")
        db.close()
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


def test_table_creation():
    """테이블 생성 테스트"""
    print("\n=== Test 2: Table Creation ===")
    
    try:
        db = TimeseriesDB(
            host='localhost',
            port=5432,
            database='bridge_monitoring_test',
            user='postgres',
            password='postgres'
        )
        
        # 테이블 생성
        db.create_tables(drop_existing=True)
        print("✓ Tables created successfully")
        
        db.close()
        return True
    except Exception as e:
        print(f"✗ Table creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_insert_and_query():
    """데이터 삽입 및 조회 테스트"""
    print("\n=== Test 3: Insert and Query ===")
    
    try:
        db = TimeseriesDB(
            host='localhost',
            port=5432,
            database='bridge_monitoring_test',
            user='postgres',
            password='postgres'
        )
        
        # 테스트 데이터
        test_data = []
        for i in range(10):
            test_data.append({
                'sensor_id': f'TEST{i:03d}',
                'hour': datetime(2024, 1, 1, i, 0),
                'ptp': 0.01 + i * 0.001,
                'mean': 0.0001,
                'std': 0.002,
                'rms': 0.0025,
                'noise_level': 0.0005,
                'natural_freqs': [2.0 + i*0.1, 5.0, 10.0],
                'kurtosis': 3.0,
                'skewness': 0.0,
                'data_count': 360000,
                'is_valid': True,
                'validation_issues': [],
                'source_zip': 'test.zip',
                'source_bin': f'test_{i}.bin',
                'file_size': 4320000
            })
        
        # 삽입
        n_inserted = db.insert_acceleration_features(test_data, on_conflict='skip')
        print(f"✓ Inserted {n_inserted} records")
        
        # 조회
        df = db.query_features(
            sensor_id='TEST000',
            table='acceleration'
        )
        print(f"✓ Queried {len(df)} records")
        
        if len(df) > 0:
            print(f"  Sample: {df.iloc[0][['sensor_id', 'time', 'ptp', 'freq_1']]}")
        
        db.close()
        return True
    
    except Exception as e:
        print(f"✗ Insert/Query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_insert():
    """대량 삽입 테스트"""
    print("\n=== Test 4: Batch Insert ===")
    
    try:
        db = TimeseriesDB(
            host='localhost',
            port=5432,
            database='bridge_monitoring_test',
            user='postgres',
            password='postgres'
        )
        
        # 대량 데이터 생성
        import time
        n_sensors = 10
        n_hours = 100
        
        test_data = []
        for sensor_idx in range(n_sensors):
            for hour_idx in range(n_hours):
                test_data.append({
                    'sensor_id': f'SENSOR{sensor_idx:03d}',
                    'hour': datetime(2024, 1, 1) + timedelta(hours=hour_idx),
                    'ptp': 0.01,
                    'mean': 0.0,
                    'std': 0.002,
                    'rms': 0.0025,
                    'noise_level': 0.0005,
                    'natural_freqs': [2.0, 5.0, 10.0],
                    'kurtosis': 3.0,
                    'skewness': 0.0,
                    'data_count': 360000,
                    'is_valid': True,
                    'validation_issues': [],
                    'source_zip': 'batch_test.zip',
                    'source_bin': f'batch_{sensor_idx}_{hour_idx}.bin',
                    'file_size': 4320000
                })
        
        print(f"Inserting {len(test_data)} records...")
        start = time.time()
        n_inserted = db.insert_acceleration_features(test_data, on_conflict='skip')
        elapsed = time.time() - start
        
        print(f"✓ Inserted {n_inserted} records in {elapsed:.2f}s")
        print(f"  Speed: {n_inserted/elapsed:.0f} records/sec")
        
        db.close()
        return True
    
    except Exception as e:
        print(f"✗ Batch insert failed: {e}")
        return False


def test_conflict_handling():
    """중복 처리 테스트"""
    print("\n=== Test 5: Conflict Handling ===")
    
    try:
        db = TimeseriesDB(
            host='localhost',
            port=5432,
            database='bridge_monitoring_test',
            user='postgres',
            password='postgres'
        )
        
        # 동일한 데이터 삽입
        test_data = [{
            'sensor_id': 'CONFLICT_TEST',
            'hour': datetime(2024, 1, 1, 0, 0),
            'ptp': 0.01,
            'mean': 0.0,
            'std': 0.002,
            'rms': 0.0025,
            'noise_level': 0.0005,
            'natural_freqs': [2.0, 5.0, 10.0],
            'kurtosis': 3.0,
            'skewness': 0.0,
            'data_count': 360000,
            'is_valid': True,
            'validation_issues': [],
            'source_zip': 'test.zip',
            'source_bin': 'test.bin',
            'file_size': 4320000
        }]
        
        # 첫 번째 삽입
        n1 = db.insert_acceleration_features(test_data, on_conflict='skip')
        print(f"✓ First insert: {n1} records")
        
        # 두 번째 삽입 (skip)
        n2 = db.insert_acceleration_features(test_data, on_conflict='skip')
        print(f"✓ Second insert (skip): {n2} records")
        
        # 세 번째 삽입 (update)
        test_data[0]['ptp'] = 0.02  # 값 변경
        n3 = db.insert_acceleration_features(test_data, on_conflict='update')
        print(f"✓ Third insert (update): {n3} records")
        
        # 확인
        df = db.query_features(sensor_id='CONFLICT_TEST')
        if len(df) > 0:
            print(f"  Final PTP value: {df.iloc[0]['ptp']}")
        
        db.close()
        return True
    
    except Exception as e:
        print(f"✗ Conflict handling failed: {e}")
        return False


def test_statistics():
    """통계 조회 테스트"""
    print("\n=== Test 6: Statistics ===")
    
    try:
        db = TimeseriesDB(
            host='localhost',
            port=5432,
            database='bridge_monitoring_test',
            user='postgres',
            password='postgres'
        )
        
        # 센서 목록
        sensors = db.get_sensor_list(table='acceleration')
        print(f"✓ Found {len(sensors)} sensors")
        
        if sensors:
            # 통계
            sensor_id = sensors[0]
            stats = db.get_statistics(sensor_id, table='acceleration')
            
            print(f"\nStatistics for {sensor_id}:")
            print(f"  Records: {stats.get('n_records', 0)}")
            print(f"  Valid rate: {stats.get('valid_rate', 0)*100:.1f}%")
            print(f"  PTP mean: {stats.get('ptp_mean', 0):.6f}")
            print(f"  Noise mean: {stats.get('noise_level_mean', 0):.6f}")
        
        db.close()
        return True
    
    except Exception as e:
        print(f"✗ Statistics failed: {e}")
        return False


def test_export():
    """내보내기 테스트"""
    print("\n=== Test 7: Export to CSV ===")
    
    try:
        db = TimeseriesDB(
            host='localhost',
            port=5432,
            database='bridge_monitoring_test',
            user='postgres',
            password='postgres'
        )
        
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / 'test_export.csv'
        
        db.export_to_csv(
            output_path=output_file,
            table='acceleration',
            limit=100
        )
        
        print(f"✓ Exported to {output_file}")
        
        # 확인
        df = pd.read_csv(output_file)
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        db.close()
        return True
    
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("Database Tests")
    print("="*60)
    print("\nNote: These tests require a running PostgreSQL/TimescaleDB instance")
    print("Database: bridge_monitoring_test")
    print("="*60)
    
    results = []
    
    # 실행할 테스트
    results.append(("Connection", test_connection()))
    results.append(("Table Creation", test_table_creation()))
    results.append(("Insert & Query", test_insert_and_query()))
    results.append(("Batch Insert", test_batch_insert()))
    results.append(("Conflict Handling", test_conflict_handling()))
    results.append(("Statistics", test_statistics()))
    results.append(("Export", test_export()))
    
    # 결과 요약
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20s} : {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)