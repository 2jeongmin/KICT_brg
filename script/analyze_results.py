"""
처리 결과 분석 및 리포트 생성
- DB 통계 조회
- 센서별 집계
- Excel 리포트 생성
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import yaml

import pandas as pd
from sqlalchemy import text

from src.database import TimeseriesDB


def load_config(config_path: Path) -> Dict:
    """설정 파일 로드"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_statistics(db: TimeseriesDB):
    """DB 통계 출력"""
    print("\n" + "=" * 80)
    print("Database Statistics")
    print("=" * 80)
    
    # 전체 레코드 수
    query = "SELECT COUNT(*) as total FROM acc_hourly_features;"
    result = db.engine.execute(text(query)).fetchone()
    print(f"Total records: {result[0]:,}")
    
    # 센서별 레코드 수
    query = """
        SELECT sensor_id, COUNT(*) as count
        FROM acc_hourly_features
        GROUP BY sensor_id
        ORDER BY sensor_id;
    """
    df = pd.read_sql(query, db.engine)
    print(f"\nTotal sensors: {len(df)}")
    print("\nTop 10 sensors by record count:")
    print(df.head(10).to_string(index=False))
    
    # 시간 범위
    query = """
        SELECT 
            MIN(time) as earliest,
            MAX(time) as latest
        FROM acc_hourly_features;
    """
    result = db.engine.execute(text(query)).fetchone()
    print(f"\nTime range:")
    print(f"  Earliest: {result[0]}")
    print(f"  Latest: {result[1]}")
    
    # 검증 통계
    query = """
        SELECT 
            COUNT(*) FILTER (WHERE is_valid = true) as valid,
            COUNT(*) FILTER (WHERE is_valid = false) as invalid
        FROM acc_hourly_features;
    """
    result = db.engine.execute(text(query)).fetchone()
    total = result[0] + result[1]
    valid_pct = result[0] / total * 100 if total > 0 else 0
    print(f"\nValidation:")
    print(f"  Valid: {result[0]:,} ({valid_pct:.1f}%)")
    print(f"  Invalid: {result[1]:,} ({100-valid_pct:.1f}%)")
    
    # 고유진동수 통계 (NULL 제외)
    query = """
        SELECT 
            AVG(freq_1) as avg_freq1,
            MIN(freq_1) as min_freq1,
            MAX(freq_1) as max_freq1
        FROM acc_hourly_features
        WHERE freq_1 IS NOT NULL;
    """
    result = db.engine.execute(text(query)).fetchone()
    if result[0] is not None:
        print(f"\nNatural Frequency (1st mode):")
        print(f"  Average: {result[0]:.3f} Hz")
        print(f"  Range: {result[1]:.3f} - {result[2]:.3f} Hz")


def generate_sensor_summary(db: TimeseriesDB) -> pd.DataFrame:
    """센서별 요약 통계"""
    query = """
        SELECT 
            sensor_id,
            COUNT(*) as record_count,
            MIN(time) as start_time,
            MAX(time) as end_time,
            AVG(ptp) as avg_ptp,
            AVG(noise_level) as avg_noise,
            AVG(freq_1) as avg_freq1,
            AVG(freq_2) as avg_freq2,
            AVG(freq_3) as avg_freq3,
            COUNT(*) FILTER (WHERE is_valid = true) as valid_count,
            COUNT(*) FILTER (WHERE is_valid = false) as invalid_count
        FROM acc_hourly_features
        GROUP BY sensor_id
        ORDER BY sensor_id;
    """
    return pd.read_sql(query, db.engine)


def generate_daily_summary(db: TimeseriesDB) -> pd.DataFrame:
    """일별 요약 통계"""
    query = """
        SELECT 
            DATE(time) as date,
            COUNT(*) as record_count,
            COUNT(DISTINCT sensor_id) as sensor_count,
            AVG(ptp) as avg_ptp,
            AVG(noise_level) as avg_noise,
            COUNT(*) FILTER (WHERE is_valid = true) as valid_count
        FROM acc_hourly_features
        GROUP BY DATE(time)
        ORDER BY date DESC
        LIMIT 100;
    """
    return pd.read_sql(query, db.engine)


def generate_quality_report(db: TimeseriesDB) -> pd.DataFrame:
    """품질 리포트"""
    query = """
        SELECT 
            sensor_id,
            COUNT(*) as total_records,
            COUNT(*) FILTER (WHERE is_valid = true) as valid_records,
            ROUND(COUNT(*) FILTER (WHERE is_valid = true)::numeric / 
                  NULLIF(COUNT(*), 0) * 100, 2) as valid_percentage,
            COUNT(*) FILTER (WHERE ptp IS NULL) as missing_ptp,
            COUNT(*) FILTER (WHERE freq_1 IS NULL) as missing_freq
        FROM acc_hourly_features
        GROUP BY sensor_id
        ORDER BY valid_percentage ASC;
    """
    return pd.read_sql(query, db.engine)


def export_to_excel(db: TimeseriesDB, output_path: Path):
    """Excel 리포트 생성"""
    print(f"\nGenerating Excel report: {output_path}")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 시트 1: 센서별 요약
        df_sensor = generate_sensor_summary(db)
        df_sensor.to_excel(writer, sheet_name='Sensor Summary', index=False)
        
        # 시트 2: 일별 요약
        df_daily = generate_daily_summary(db)
        df_daily.to_excel(writer, sheet_name='Daily Summary', index=False)
        
        # 시트 3: 품질 리포트
        df_quality = generate_quality_report(db)
        df_quality.to_excel(writer, sheet_name='Quality Report', index=False)
        
        # 시트 4: 전체 통계 (텍스트)
        stats_data = {
            'Metric': [
                'Total Records',
                'Total Sensors',
                'Valid Records',
                'Invalid Records',
                'Data Completeness (%)'
            ],
            'Value': []
        }
        
        query = "SELECT COUNT(*) FROM acc_hourly_features;"
        total = db.engine.execute(text(query)).scalar()
        stats_data['Value'].append(total)
        
        query = "SELECT COUNT(DISTINCT sensor_id) FROM acc_hourly_features;"
        sensors = db.engine.execute(text(query)).scalar()
        stats_data['Value'].append(sensors)
        
        query = "SELECT COUNT(*) FROM acc_hourly_features WHERE is_valid = true;"
        valid = db.engine.execute(text(query)).scalar()
        stats_data['Value'].append(valid)
        
        invalid = total - valid
        stats_data['Value'].append(invalid)
        
        completeness = valid / total * 100 if total > 0 else 0
        stats_data['Value'].append(f"{completeness:.2f}")
        
        pd.DataFrame(stats_data).to_excel(writer, sheet_name='Overall Stats', index=False)
    
    print(f"✅ Excel report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze processing results and generate reports"
    )
    parser.add_argument(
        '-c', '--config',
        type=Path,
        default=Path('config/default.yaml'),
        help="Config file path"
    )
    parser.add_argument(
        '-d', '--database',
        action='store_true',
        help="Show database statistics"
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help="Excel output path (e.g., output/report.xlsx)"
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # DB 연결
    db_config = config['database']
    db = TimeseriesDB(
        host=db_config['host'],
        port=db_config.get('port', 5432),
        database=db_config['database'],
        user=db_config['user'],
        password=db_config['password']
    )
    
    try:
        # DB 통계 출력
        if args.database:
            print_statistics(db)
        
        # Excel 리포트 생성
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            export_to_excel(db, args.output)
        
        # 옵션 없으면 도움말
        if not args.database and not args.output:
            parser.print_help()
            print("\nExample usage:")
            print("  python scripts/analyze_results.py -d")
            print("  python scripts/analyze_results.py -d -o output/report.xlsx")
    
    finally:
        db.close()


if __name__ == "__main__":
    main()