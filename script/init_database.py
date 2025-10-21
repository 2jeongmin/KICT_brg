#!/usr/bin/env python3
# scripts/init_database.py
"""
데이터베이스 초기화 스크립트
테이블 생성 및 기본 설정
"""

import sys
from pathlib import Path
import yaml
import argparse

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database import TimeseriesDB
import logging

logging.basicConfig(
    level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Initialize TimescaleDB database')
    parser.add_argument('--config', '-c', type=Path, default=Path('config/default.yaml'),
                       help='Config file path')
    parser.add_argument('--drop', action='store_true',
                       help='Drop existing tables')
    parser.add_argument('--host', type=str, help='Database host')
    parser.add_argument('--port', type=int, help='Database port')
    parser.add_argument('--database', type=str, help='Database name')
    parser.add_argument('--user', type=str, help='Database user')
    parser.add_argument('--password', type=str, help='Database password')
    
    args = parser.parse_args()
    
    # 설정 로드
    if args.config.exists():
        config = load_config(args.config)
        db_config = config.get('database', {})
    else:
        logger.warning(f"Config file not found: {args.config}")
        db_config = {}
    
    # 커맨드라인 인자로 오버라이드
    if args.host:
        db_config['host'] = args.host
    if args.port:
        db_config['port'] = args.port
    if args.database:
        db_config['database'] = args.database
    if args.user:
        db_config['user'] = args.user
    if args.password:
        db_config['password'] = args.password
    
    logger.info("="*60)
    logger.info("Database Initialization")
    logger.info("="*60)
    logger.info(f"Host: {db_config.get('host', 'localhost')}")
    logger.info(f"Port: {db_config.get('port', 5432)}")
    logger.info(f"Database: {db_config.get('database', 'bridge_monitoring')}")
    logger.info(f"User: {db_config.get('user', 'postgres')}")
    logger.info(f"Drop existing: {args.drop}")
    logger.info("="*60)
    
    try:
        # DB 연결
        db = TimeseriesDB(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            database=db_config.get('database', 'bridge_monitoring'),
            user=db_config.get('user', 'postgres'),
            password=db_config.get('password', 'postgres'),
            schema=db_config.get('schema', 'public')
        )
        
        # 테이블 생성
        logger.info("Creating tables...")
        db.create_tables(drop_existing=args.drop)
        
        # 생성된 테이블 확인
        logger.info("\nVerifying tables...")
        sensors = db.get_sensor_list(table='acceleration')
        logger.info(f"Acceleration table: OK (sensors: {len(sensors)})")
        
        sensors = db.get_sensor_list(table='static')
        logger.info(f"Static table: OK (sensors: {len(sensors)})")
        
        logger.info("\n" + "="*60)
        logger.info("Database initialization completed successfully!")
        logger.info("="*60)
        
        db.close()
        return 0
    
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())