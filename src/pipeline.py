# src/pipeline.py
"""
전체 처리 파이프라인
Zip 파일 → 특성값 추출 → DB 저장 전체 프로세스 통합

주요 기능:
- 모든 모듈 통합 및 조율
- 진행률 추적 및 로깅
- 에러 핸들링 및 복구
- 체크포인트 및 재시작
- 성능 모니터링
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from tqdm import tqdm

from .zip_handler import ZipHandler
from .feature_extraction import FeatureExtractor, BatchFeatureExtractor
from .database import TimeseriesDB

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """파이프라인 상태 추적"""
    start_time: datetime
    last_update: datetime
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_records: int = 0
    saved_records: int = 0
    failed_records: int = 0
    current_batch: int = 0
    last_processed_file: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        d = asdict(self)
        d['start_time'] = self.start_time.isoformat()
        d['last_update'] = self.last_update.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PipelineState':
        """딕셔너리에서 생성"""
        d['start_time'] = datetime.fromisoformat(d['start_time'])
        d['last_update'] = datetime.fromisoformat(d['last_update'])
        return cls(**d)
    
    def update_progress(self, 
                       processed: int = 0,
                       failed: int = 0,
                       records: int = 0,
                       saved: int = 0):
        """진행률 업데이트"""
        self.processed_files += processed
        self.failed_files += failed
        self.total_records += records
        self.saved_records += saved
        self.last_update = datetime.now()


class ProcessingPipeline:
    """
    전체 처리 파이프라인 클래스
    """
    
    def __init__(self, 
                 config: Dict,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            config: 설정 딕셔너리
            logger: 로거 인스턴스
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # 설정 파싱
        self.data_config = config.get('data', {})
        self.features_config = config.get('features', {})
        self.db_config = config.get('database', {})
        self.proc_config = config.get('processing', {})
        self.monitor_config = config.get('monitoring', {})
        
        # 모듈 초기화
        self._init_modules()
        
        # 상태 관리
        self.state = None
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 통계
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_duration': 0,
            'files_per_second': 0,
            'records_per_second': 0,
            'errors': defaultdict(int)
        }
    
    
    def _init_modules(self):
        """모듈 초기화"""
        self.logger.info("Initializing modules...")
        
        # 1. Zip Handler
        self.zip_handler = ZipHandler(
            mode=self.data_config.get('mode', 'stream'),
            temp_dir=self.data_config.get('temp_dir')
        )
        
        # 2. Feature Extractor
        acc_config = self.features_config.get('acceleration', {})
        fft_config = acc_config.get('fft', {})
        noise_config = acc_config.get('noise', {})
        
        self.feature_extractor = FeatureExtractor(
            sampling_rate=acc_config.get('sampling_rate', 100),
            fft_freq_range=tuple(fft_config.get('freq_range', [0.5, 20])),
            fft_n_peaks=fft_config.get('n_peaks', 3),
            fft_prominence_ratio=fft_config.get('prominence_ratio', 0.1),
            noise_method=noise_config.get('method', 'static_std'),
            noise_static_threshold=noise_config.get('static_threshold', 0.01)
        )
        
        self.batch_extractor = BatchFeatureExtractor(self.feature_extractor)
        
        # 3. Database
        self.db = TimeseriesDB(
            host=self.db_config.get('host', 'localhost'),
            port=self.db_config.get('port', 5432),
            database=self.db_config.get('database', 'bridge_monitoring'),
            user=self.db_config.get('user', 'postgres'),
            password=self.db_config.get('password', 'postgres'),
            schema=self.db_config.get('schema', 'public')
        )
        
        self.logger.info("Modules initialized successfully")
    
    
    def run(self) -> Dict:
        """
        파이프라인 실행 (메인 함수)
        
        Returns:
            실행 결과 딕셔너리
        """
        self.logger.info("="*60)
        self.logger.info("Starting Processing Pipeline")
        self.logger.info("="*60)
        
        try:
            # 1. 초기화
            self._setup()
            
            # 2. 재시작 체크
            if self._check_restart():
                self.logger.info("Resuming from checkpoint...")
            
            # 3. 파일 목록 생성
            zip_files = self._get_file_list()
            
            # 4. 처리
            self._process_files(zip_files)
            
            # 5. 마무리
            results = self._finalize()
            
            self.logger.info("="*60)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info("="*60)
            
            return results
        
        except KeyboardInterrupt:
            self.logger.warning("Pipeline interrupted by user")
            self._save_checkpoint()
            raise
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            self._save_checkpoint()
            raise
        
        finally:
            self._cleanup()
    
    
    def _setup(self):
        """초기 설정"""
        self.logger.info("Setting up pipeline...")
        
        # 상태 초기화
        self.state = PipelineState(
            start_time=datetime.now(),
            last_update=datetime.now()
        )
        
        self.stats['start_time'] = datetime.now()
        
        # DB 테이블 생성 (없으면)
        try:
            self.db.create_tables(drop_existing=False)
        except Exception as e:
            self.logger.debug(f"Table creation skipped: {e}")
    
    
    def _get_file_list(self) -> List[Path]:
        """처리할 파일 목록 생성"""
        root_dir = Path(self.data_config['root_dir'])
        pattern = self.data_config.get('file_pattern', '*.zip')
        
        # 파일 찾기
        zip_files = self.zip_handler.find_zip_files(root_dir, pattern)
        
        # 필터 적용
        zip_files = self._apply_filters(zip_files)
        
        # 최대 파일 수 제한
        max_files = self.proc_config.get('max_files')
        if max_files:
            zip_files = zip_files[:max_files]
            self.logger.info(f"Limited to {max_files} files (test mode)")
        
        self.state.total_files = len(zip_files)
        self.logger.info(f"Found {len(zip_files)} files to process")
        
        return zip_files
    
    
    def _apply_filters(self, zip_files: List[Path]) -> List[Path]:
        """파일 필터 적용"""
        # 센서 필터
        target_sensors = self.proc_config.get('sensors')
        if target_sensors:
            filtered = []
            for f in zip_files:
                for sensor in target_sensors:
                    if sensor in f.name:
                        filtered.append(f)
                        break
            zip_files = filtered
            self.logger.info(f"Filtered by sensors: {len(zip_files)} files")
        
        # 날짜 범위 필터
        date_range = self.proc_config.get('date_range', {})
        start_date = date_range.get('start')
        end_date = date_range.get('end')
        
        if start_date or end_date:
            # 파일명에서 날짜 추출 필요
            # 구현 생략 (필요시 추가)
            pass
        
        # 이미 처리된 파일 건너뛰기
        if self.proc_config.get('skip_existing', True):
            zip_files = self._filter_existing(zip_files)
        
        return zip_files
    
    
    def _filter_existing(self, zip_files: List[Path]) -> List[Path]:
        """이미 처리된 파일 제외"""
        # 구현: DB에서 이미 처리된 파일 체크
        # 간단하게 구현 (실제로는 더 정교한 로직 필요)
        
        # 파일명에서 센서ID와 시간 추출 후 DB 체크
        # 여기서는 생략하고 모든 파일 반환
        
        return zip_files
    
    
    def _process_files(self, zip_files: List[Path]):
        """파일 처리 메인 루프"""
        batch_size = self.proc_config.get('batch_size', 100)
        report_interval = self.monitor_config.get('report_interval', 10)
        
        batch_features = []
        
        # 진행률 표시
        pbar = tqdm(
            zip_files,
            desc="Processing files",
            unit="file",
            initial=self.state.processed_files
        )
        
        for zip_file in pbar:
            try:
                # Zip 파일 처리
                zip_results = self.zip_handler.process_single_zip(zip_file)
                
                if not zip_results:
                    self.state.update_progress(failed=1)
                    self.stats['errors']['empty_zip'] += 1
                    continue
                
                # 특성값 추출
                features_list = self.batch_extractor.extract_from_zip_results(
                    zip_results,
                    sensor_type='acceleration'
                )
                
                batch_features.extend(features_list)
                
                # 배치 크기 도달 시 DB 저장
                if len(batch_features) >= batch_size:
                    saved = self._save_batch(batch_features)
                    
                    self.state.update_progress(
                        processed=len(zip_results),
                        records=len(batch_features),
                        saved=saved
                    )
                    
                    batch_features = []
                    self.state.current_batch += 1
                
                # 진행 상황 업데이트
                self.state.last_processed_file = zip_file.name
                
                # 주기적 리포트
                if self.state.processed_files % report_interval == 0:
                    self._report_progress(pbar)
                
                # 주기적 체크포인트 저장
                if self.state.processed_files % 100 == 0:
                    self._save_checkpoint()
            
            except Exception as e:
                self.logger.error(f"Failed to process {zip_file.name}: {e}")
                self.state.update_progress(failed=1)
                self.stats['errors']['processing_error'] += 1
                continue
        
        # 남은 배치 저장
        if batch_features:
            saved = self._save_batch(batch_features)
            self.state.update_progress(
                records=len(batch_features),
                saved=saved
            )
        
        pbar.close()
    
    
    def _save_batch(self, features_list: List[Dict]) -> int:
        """배치 저장"""
        try:
            on_conflict = self.db_config.get('on_conflict', 'skip')
            
            if self.proc_config.get('reprocess', False):
                on_conflict = 'update'
            
            n_saved = self.db.insert_acceleration_features(
                features_list,
                on_conflict=on_conflict
            )
            
            return n_saved
        
        except Exception as e:
            self.logger.error(f"Failed to save batch: {e}")
            self.stats['errors']['db_error'] += 1
            return 0
    
    
    def _report_progress(self, pbar: Optional[tqdm] = None):
        """진행 상황 리포트"""
        elapsed = (datetime.now() - self.state.start_time).total_seconds()
        
        if elapsed > 0:
            files_per_sec = self.state.processed_files / elapsed
            records_per_sec = self.state.saved_records / elapsed
            
            # 남은 시간 예측
            remaining_files = self.state.total_files - self.state.processed_files
            if files_per_sec > 0:
                eta_seconds = remaining_files / files_per_sec
                eta = timedelta(seconds=int(eta_seconds))
            else:
                eta = "Unknown"
            
            # 진행률
            progress = (self.state.processed_files / self.state.total_files * 100) if self.state.total_files > 0 else 0
            
            report = {
                'progress': f"{progress:.1f}%",
                'processed': f"{self.state.processed_files}/{self.state.total_files}",
                'failed': self.state.failed_files,
                'records': self.state.saved_records,
                'speed': f"{files_per_sec:.2f} files/s, {records_per_sec:.0f} records/s",
                'eta': str(eta)
            }
            
            if pbar:
                pbar.set_postfix(report)
            
            self.logger.info(f"Progress: {report}")
    
    
    def _save_checkpoint(self):
        """체크포인트 저장"""
        checkpoint_file = self.checkpoint_dir / 'pipeline_checkpoint.json'
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
            
            self.logger.debug(f"Checkpoint saved: {checkpoint_file}")
        
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    
    def _check_restart(self) -> bool:
        """체크포인트에서 재시작"""
        checkpoint_file = self.checkpoint_dir / 'pipeline_checkpoint.json'
        
        if not checkpoint_file.exists():
            return False
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # 재시작 확인 (1시간 이내 중단만)
            last_update = datetime.fromisoformat(checkpoint_data['last_update'])
            if (datetime.now() - last_update).total_seconds() > 3600:
                self.logger.info("Checkpoint is too old, starting fresh")
                return False
            
            self.state = PipelineState.from_dict(checkpoint_data)
            self.logger.info(f"Resumed from checkpoint: {checkpoint_data['processed_files']} files processed")
            
            return True
        
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return False
    
    
    def _finalize(self) -> Dict:
        """마무리 및 결과 생성"""
        self.stats['end_time'] = datetime.now()
        self.stats['total_duration'] = (
            self.stats['end_time'] - self.stats['start_time']
        ).total_seconds()
        
        # 속도 계산
        if self.stats['total_duration'] > 0:
            self.stats['files_per_second'] = (
                self.state.processed_files / self.stats['total_duration']
            )
            self.stats['records_per_second'] = (
                self.state.saved_records / self.stats['total_duration']
            )
        
        # 결과 요약
        results = {
            'success': True,
            'start_time': self.stats['start_time'].isoformat(),
            'end_time': self.stats['end_time'].isoformat(),
            'duration_seconds': self.stats['total_duration'],
            'duration_formatted': str(timedelta(seconds=int(self.stats['total_duration']))),
            'files': {
                'total': self.state.total_files,
                'processed': self.state.processed_files,
                'failed': self.state.failed_files,
                'success_rate': (
                    self.state.processed_files / self.state.total_files * 100
                    if self.state.total_files > 0 else 0
                )
            },
            'records': {
                'total': self.state.total_records,
                'saved': self.state.saved_records,
                'failed': self.state.failed_records
            },
            'performance': {
                'files_per_second': self.stats['files_per_second'],
                'records_per_second': self.stats['records_per_second']
            },
            'errors': dict(self.stats['errors'])
        }
        
        # 로그 출력
        self._log_summary(results)
        
        # 결과 저장
        self._save_results(results)
        
        return results
    
    
    def _log_summary(self, results: Dict):
        """결과 요약 로그"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Processing Summary")
        self.logger.info("="*60)
        
        self.logger.info(f"Duration: {results['duration_formatted']}")
        self.logger.info(f"Files processed: {results['files']['processed']}/{results['files']['total']}")
        self.logger.info(f"Files failed: {results['files']['failed']}")
        self.logger.info(f"Success rate: {results['files']['success_rate']:.1f}%")
        
        self.logger.info(f"\nRecords saved: {results['records']['saved']}")
        self.logger.info(f"Records failed: {results['records']['failed']}")
        
        self.logger.info(f"\nPerformance:")
        self.logger.info(f"  {results['performance']['files_per_second']:.2f} files/sec")
        self.logger.info(f"  {results['performance']['records_per_second']:.0f} records/sec")
        
        if results['errors']:
            self.logger.info(f"\nErrors:")
            for error_type, count in results['errors'].items():
                self.logger.info(f"  {error_type}: {count}")
        
        self.logger.info("="*60)
    
    
    def _save_results(self, results: Dict):
        """결과를 파일로 저장"""
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = output_dir / f'pipeline_results_{timestamp}.json'
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to: {result_file}")
    
    
    def _cleanup(self):
        """정리 작업"""
        try:
            # DB 연결 종료
            self.db.close()
            
            # 체크포인트 삭제 (정상 완료 시)
            if self.state and self.state.processed_files == self.state.total_files:
                checkpoint_file = self.checkpoint_dir / 'pipeline_checkpoint.json'
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                    self.logger.debug("Checkpoint removed")
        
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")
    
    
    def get_database_statistics(self) -> Dict:
        """DB 통계 조회"""
        sensors = self.db.get_sensor_list(table='acceleration')
        
        stats = {
            'total_sensors': len(sensors),
            'sensors': []
        }
        
        # 각 센서별 통계 (최대 10개)
        for sensor_id in sensors[:10]:
            sensor_stats = self.db.get_statistics(
                sensor_id,
                table='acceleration'
            )
            stats['sensors'].append(sensor_stats)
        
        return stats


# 편의 함수

def run_pipeline_from_config(config_path: Path) -> Dict:
    """
    설정 파일로 파이프라인 실행
    
    Args:
        config_path: 설정 파일 경로
    
    Returns:
        실행 결과
    """
    import yaml
    
    # 설정 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 파이프라인 실행
    pipeline = ProcessingPipeline(config)
    results = pipeline.run()
    
    return results


if __name__ == "__main__":
    """
    테스트 실행
    """
    import sys
    import yaml
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <config_file>")
        print("\nExample:")
        print("  python pipeline.py config/test.yaml")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # 실행
    try:
        results = run_pipeline_from_config(config_path)
        print("\n" + "="*60)
        print("Pipeline completed!")
        print(f"Processed: {results['files']['processed']} files")
        print(f"Saved: {results['records']['saved']} records")
        print("="*60)
        sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)