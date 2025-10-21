#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
교량 센서 데이터 특성값 추출 시스템 - 메인 실행 파일

Usage:
    python main.py --config config/default.yaml
    python main.py --config config/test.yaml --max-files 10
    python main.py --config config/default.yaml --sensors DNA21001,DNA21002
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import yaml
import logging
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_logging(config: dict) -> logging.Logger:
    """
    로깅 설정
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        Logger 객체
    """
    log_config = config.get('logging', {})
    
    # 로거 생성
    logger = logging.getLogger('bridge_monitoring')
    logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
    
    # 기존 핸들러 제거
    logger.handlers.clear()
    
    # 포맷터
    formatter = logging.Formatter(
        log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    )
    
    # 콘솔 핸들러
    if log_config.get('console', {}).get('enable', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 파일 핸들러
    if log_config.get('file', {}).get('enable', True):
        log_dir = Path(log_config['file'].get('dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        # 파일명 생성
        timestamp = datetime.now().strftime(
            log_config['file'].get('filename_format', 'processing_%Y%m%d_%H%M%S.log')
        )
        log_file = log_dir / timestamp
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Log file: {log_file}")
    
    return logger


def load_config(config_path: Path) -> dict:
    """
    YAML 설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
    
    Returns:
        설정 딕셔너리
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def override_config_from_args(config: dict, args: argparse.Namespace) -> dict:
    """
    커맨드라인 인자로 설정 오버라이드
    
    Args:
        config: 기본 설정
        args: 커맨드라인 인자
    
    Returns:
        오버라이드된 설정
    """
    # max_files
    if args.max_files is not None:
        config['processing']['max_files'] = args.max_files
    
    # sensors
    if args.sensors:
        config['processing']['sensors'] = args.sensors.split(',')
    
    # workers
    if args.workers is not None:
        config['processing']['n_workers'] = args.workers
    
    # reprocess
    if args.reprocess:
        config['processing']['reprocess'] = True
    
    # features
    if args.features:
        feature_list = args.features.split(',')
        config['features']['acceleration']['extract'] = feature_list
    
    # date range
    if args.start_date:
        config['processing']['date_range']['start'] = args.start_date
    if args.end_date:
        config['processing']['date_range']['end'] = args.end_date
    
    return config


def validate_config(config: dict, logger: logging.Logger):
    """
    설정 유효성 검사
    
    Args:
        config: 설정 딕셔너리
        logger: 로거
    """
    # 필수 항목 체크
    required_keys = ['data', 'features', 'database', 'processing']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    # 데이터 디렉토리 존재 확인
    data_dir = Path(config['data']['root_dir'])
    if not data_dir.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
    
    # 처리 모드 체크
    valid_modes = ['stream', 'extract_delete', 'temp_dir']
    mode = config['data']['mode']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
    
    logger.info("Configuration validated successfully")


def print_config_summary(config: dict, logger: logging.Logger):
    """
    설정 요약 출력
    
    Args:
        config: 설정 딕셔너리
        logger: 로거
    """
    logger.info("=" * 60)
    logger.info("Configuration Summary")
    logger.info("=" * 60)
    
    # 데이터 소스
    logger.info(f"Data Directory: {config['data']['root_dir']}")
    logger.info(f"Processing Mode: {config['data']['mode']}")
    
    # 특성값
    features = config['features']['acceleration']['extract']
    logger.info(f"Features to Extract: {', '.join(features)}")
    
    # 처리 설정
    proc = config['processing']
    logger.info(f"Batch Size: {proc['batch_size']}")
    logger.info(f"Workers: {proc['n_workers']}")
    
    if proc.get('max_files'):
        logger.info(f"Max Files: {proc['max_files']} (TEST MODE)")
    
    if proc.get('sensors'):
        logger.info(f"Target Sensors: {', '.join(proc['sensors'])}")
    
    # 데이터베이스
    db = config['database']
    logger.info(f"Database: {db['user']}@{db['host']}:{db['port']}/{db['database']}")
    
    logger.info("=" * 60)


def main():
    """
    메인 실행 함수
    """
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(
        description='교량 센서 데이터 특성값 추출 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 실행
  python main.py --config config/default.yaml
  
  # 테스트 모드 (10개 파일만)
  python main.py --config config/test.yaml --max-files 10
  
  # 특정 센서만 처리
  python main.py --config config/default.yaml --sensors DNA21001,DNA21002
  
  # 특정 특성값만 추출
  python main.py --config config/default.yaml --features ptp,noise,fft
  
  # 병렬 처리 워커 수 지정
  python main.py --config config/default.yaml --workers 8
  
  # 재처리 (기존 데이터 덮어쓰기)
  python main.py --config config/default.yaml --reprocess
        """
    )
    
    # 필수 인자
    parser.add_argument(
        '--config', '-c',
        type=Path,
        required=True,
        help='설정 파일 경로 (YAML)'
    )
    
    # 선택 인자
    parser.add_argument(
        '--max-files',
        type=int,
        help='처리할 최대 파일 수 (테스트용)'
    )
    
    parser.add_argument(
        '--sensors',
        type=str,
        help='처리할 센서 ID (쉼표 구분, 예: DNA21001,DNA21002)'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        help='추출할 특성값 (쉼표 구분, 예: ptp,noise,fft)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        help='병렬 처리 워커 수'
    )
    
    parser.add_argument(
        '--reprocess',
        action='store_true',
        help='기존 데이터 재처리 (덮어쓰기)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='시작 날짜 (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='종료 날짜 (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제 처리 없이 설정만 확인'
    )
    
    args = parser.parse_args()
    
    try:
        # 1. 설정 로드
        config = load_config(args.config)
        
        # 2. 커맨드라인 인자로 오버라이드
        config = override_config_from_args(config, args)
        
        # 3. 로깅 설정
        logger = setup_logging(config)
        
        logger.info("=" * 60)
        logger.info("Bridge Monitoring Data Processing System")
        logger.info("=" * 60)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 4. 설정 검증
        validate_config(config, logger)
        
        # 5. 설정 요약 출력
        print_config_summary(config, logger)
        
        # 6. Dry-run 모드
        if args.dry_run:
            logger.info("DRY-RUN mode: Configuration validated. Exiting.")
            return 0
        
        # 7. 파이프라인 실행
        logger.info("Starting data processing...")
        
        from src.pipeline import ProcessingPipeline
        
        pipeline = ProcessingPipeline(config, logger)
        results = pipeline.run()
        
        # 결과 출력
        logger.info("\n" + "="*60)
        logger.info("Final Results")
        logger.info("="*60)
        logger.info(f"Files processed: {results['files']['processed']}/{results['files']['total']}")
        logger.info(f"Records saved: {results['records']['saved']}")
        logger.info(f"Duration: {results['duration_formatted']}")
        logger.info(f"Performance: {results['performance']['files_per_second']:.2f} files/sec")
        
        # DB 통계
        logger.info("\nDatabase Statistics:")
        db_stats = pipeline.get_database_statistics()
        logger.info(f"Total sensors: {db_stats['total_sensors']}")
        
        logger.info("="*60)
        logger.info("Processing completed successfully!")
        logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())