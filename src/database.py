# src/database.py
"""
TimescaleDB 데이터베이스 연동 모듈

주요 기능:
- 테이블 스키마 생성 (Hypertable)
- 특성값 배치 저장
- 데이터 조회 및 쿼리
- 중복 체크 및 재처리 지원
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
from pandas import DataFrame
import numpy as np
from sqlalchemy import (
    create_engine, 
    Column, 
    String, 
    Float, 
    Integer, 
    Boolean, 
    DateTime,
    ARRAY,
    MetaData,
    Table,
    text,
    select,
    and_,
    or_
)
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.postgresql import insert
import logging

logger = logging.getLogger(__name__)


class TimeseriesDB:
    """
    TimescaleDB 데이터베이스 연동 클래스
    """
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 5432,
                 database: str = 'bridge_monitoring',
                 user: str = 'postgres',
                 password: str = 'postgres',
                 schema: str = 'public',
                 pool_size: int = 5,
                 max_overflow: int = 10):
        """
        Args:
            host: DB 호스트
            port: DB 포트
            database: 데이터베이스 이름
            user: 사용자명
            password: 비밀번호
            schema: 스키마 이름
            pool_size: 커넥션 풀 크기
            max_overflow: 최대 오버플로우
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.schema = schema
        
        # 연결 문자열
        self.connection_string = (
            f"postgresql://{user}:{password}@{host}:{port}/{database}"
        )
        
        # SQLAlchemy 엔진 생성
        self.engine = create_engine(
            self.connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # 연결 유효성 체크
            echo=False
        )
        
        # 세션 팩토리
        self.Session = sessionmaker(bind=self.engine)
        
        # 메타데이터
        self.metadata = MetaData(schema=schema)
        
        # 테이블 정의
        self._define_tables()
        
        logger.info(f"Connected to database: {host}:{port}/{database}")
    
    
    def _define_tables(self):
        """
        테이블 스키마 정의
        """
        # 가속도 센서 특성값 테이블
        self.acc_features_table = Table(
            'acc_hourly_features',
            self.metadata,
            Column('time', DateTime(timezone=True), primary_key=True, nullable=False),
            Column('sensor_id', String(50), primary_key=True, nullable=False),
            Column('ptp', Float),
            Column('mean', Float),
            Column('std', Float),
            Column('min', Float),
            Column('max', Float),
            Column('median', Float),
            Column('rms', Float),
            Column('noise_level', Float),
            Column('freq_1', Float),
            Column('freq_2', Float),
            Column('freq_3', Float),
            Column('kurtosis', Float),
            Column('skewness', Float),
            Column('data_count', Integer),
            Column('is_valid', Boolean),
            Column('validation_issues', String(500)),
            Column('source_zip', String(255)),
            Column('source_bin', String(255)),
            Column('file_size', Integer),
            Column('processed_at', DateTime(timezone=True), default=datetime.utcnow),
            schema=self.schema
        )
        
        # 정적 센서 특성값 테이블
        self.static_features_table = Table(
            'static_hourly_features',
            self.metadata,
            Column('time', DateTime(timezone=True), primary_key=True, nullable=False),
            Column('sensor_id', String(50), primary_key=True, nullable=False),
            Column('sensor_type', String(50)),  # temperature, humidity, displacement, crack
            Column('mean', Float),
            Column('ptp', Float),
            Column('noise_level', Float),
            Column('min', Float),
            Column('max', Float),
            Column('data_count', Integer),
            Column('is_valid', Boolean),
            Column('source_file', String(255)),
            Column('processed_at', DateTime(timezone=True), default=datetime.utcnow),
            schema=self.schema
        )
    
    
    def create_tables(self, drop_existing: bool = False):
        """
        테이블 생성
        
        Args:
            drop_existing: 기존 테이블 삭제 여부
        """
        with self.engine.begin() as conn:
            if drop_existing:
                logger.warning("Dropping existing tables...")
                self.metadata.drop_all(conn)
            
            # 테이블 생성
            self.metadata.create_all(conn)
            logger.info("Tables created successfully")
            
            # TimescaleDB Hypertable 생성
            self._create_hypertables(conn)
            
            # 인덱스 생성
            self._create_indexes(conn)
    
    
    def _create_hypertables(self, conn):
        """
        TimescaleDB Hypertable 생성
        
        Args:
            conn: DB 연결
        """
        try:
            # 가속도 테이블
            conn.execute(text(
                f"""
                SELECT create_hypertable(
                    '{self.schema}.acc_hourly_features',
                    'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 month'
                );
                """
            ))
            logger.info("Created hypertable: acc_hourly_features")
            
            # 정적 센서 테이블
            conn.execute(text(
                f"""
                SELECT create_hypertable(
                    '{self.schema}.static_hourly_features',
                    'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 month'
                );
                """
            ))
            logger.info("Created hypertable: static_hourly_features")
            
        except Exception as e:
            logger.warning(f"Hypertable creation warning (may already exist): {e}")
    
    
    def _create_indexes(self, conn):
        """
        인덱스 생성
        
        Args:
            conn: DB 연결
        """
        indexes = [
            # 가속도 테이블 인덱스
            f"CREATE INDEX IF NOT EXISTS idx_acc_sensor_time ON {self.schema}.acc_hourly_features (sensor_id, time DESC);",
            f"CREATE INDEX IF NOT EXISTS idx_acc_valid ON {self.schema}.acc_hourly_features (is_valid);",
            f"CREATE INDEX IF NOT EXISTS idx_acc_processed ON {self.schema}.acc_hourly_features (processed_at);",
            
            # 정적 센서 테이블 인덱스
            f"CREATE INDEX IF NOT EXISTS idx_static_sensor_time ON {self.schema}.static_hourly_features (sensor_id, time DESC);",
            f"CREATE INDEX IF NOT EXISTS idx_static_type ON {self.schema}.static_hourly_features (sensor_type);",
        ]
        
        for idx_sql in indexes:
            try:
                conn.execute(text(idx_sql))
            except Exception as e:
                logger.debug(f"Index creation skipped (may exist): {e}")
        
        logger.info("Indexes created successfully")
    
    
    def insert_acceleration_features(self, 
                                    features_list: List[Dict],
                                    on_conflict: str = 'skip') -> int:
        """
        가속도 센서 특성값 배치 저장
        
        Args:
            features_list: 특성값 딕셔너리 리스트
            on_conflict: 충돌 처리 ('skip', 'update')
        
        Returns:
            저장된 레코드 수
        """
        if not features_list:
            logger.warning("No features to insert")
            return 0
        
        # 데이터 전처리
        records = []
        for features in features_list:
            record = self._prepare_acceleration_record(features)
            if record:
                records.append(record)
        
        if not records:
            logger.warning("No valid records after preprocessing")
            return 0
        
        # 배치 insert
        with self.engine.begin() as conn:
            if on_conflict == 'skip':
                # 중복 시 건너뛰기
                stmt = insert(self.acc_features_table).values(records)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=['time', 'sensor_id']
                )
            
            elif on_conflict == 'update':
                # 중복 시 업데이트
                stmt = insert(self.acc_features_table).values(records)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['time', 'sensor_id'],
                    set_={
                        'ptp': stmt.excluded.ptp,
                        'mean': stmt.excluded.mean,
                        'std': stmt.excluded.std,
                        'rms': stmt.excluded.rms,
                        'noise_level': stmt.excluded.noise_level,
                        'freq_1': stmt.excluded.freq_1,
                        'freq_2': stmt.excluded.freq_2,
                        'freq_3': stmt.excluded.freq_3,
                        'kurtosis': stmt.excluded.kurtosis,
                        'skewness': stmt.excluded.skewness,
                        'data_count': stmt.excluded.data_count,
                        'is_valid': stmt.excluded.is_valid,
                        'validation_issues': stmt.excluded.validation_issues,
                        'processed_at': datetime.utcnow()
                    }
                )
            
            else:
                raise ValueError(f"Invalid on_conflict: {on_conflict}")
            
            result = conn.execute(stmt)
        
        inserted_count = len(records)
        logger.info(f"Inserted {inserted_count} acceleration feature records")
        
        return inserted_count
    
    
    def insert_static_features(self, 
                              features_list: List[Dict],
                              on_conflict: str = 'skip') -> int:
        """
        정적 센서 특성값 배치 저장
        
        Args:
            features_list: 특성값 딕셔너리 리스트
            on_conflict: 충돌 처리 ('skip', 'update')
        
        Returns:
            저장된 레코드 수
        """
        if not features_list:
            return 0
        
        records = []
        for features in features_list:
            record = self._prepare_static_record(features)
            if record:
                records.append(record)
        
        if not records:
            return 0
        
        with self.engine.begin() as conn:
            if on_conflict == 'skip':
                stmt = insert(self.static_features_table).values(records)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=['time', 'sensor_id']
                )
            elif on_conflict == 'update':
                stmt = insert(self.static_features_table).values(records)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['time', 'sensor_id'],
                    set_={
                        'mean': stmt.excluded.mean,
                        'ptp': stmt.excluded.ptp,
                        'noise_level': stmt.excluded.noise_level,
                        'data_count': stmt.excluded.data_count,
                        'is_valid': stmt.excluded.is_valid,
                        'processed_at': datetime.utcnow()
                    }
                )
            
            conn.execute(stmt)
        
        logger.info(f"Inserted {len(records)} static feature records")
        return len(records)
    
    
    def _prepare_acceleration_record(self, features: Dict) -> Optional[Dict]:
        """
        가속도 특성값을 DB 레코드로 변환
        
        Args:
            features: 특성값 딕셔너리
        
        Returns:
            DB 레코드 딕셔너리
        """
        try:
            # 필수 필드 체크
            if 'sensor_id' not in features or 'hour' not in features:
                logger.warning("Missing required fields: sensor_id or hour")
                return None
            
            # natural_freqs 배열을 개별 컬럼으로 분리
            nat_freqs = features.get('natural_freqs', [])
            freq_1 = nat_freqs[0] if len(nat_freqs) > 0 else None
            freq_2 = nat_freqs[1] if len(nat_freqs) > 1 else None
            freq_3 = nat_freqs[2] if len(nat_freqs) > 2 else None
            
            # validation_issues를 문자열로 변환
            validation_issues = features.get('validation_issues', [])
            if isinstance(validation_issues, list):
                validation_issues = ','.join(validation_issues) if validation_issues else None
            
            record = {
                'time': features['hour'],
                'sensor_id': features['sensor_id'],
                'ptp': self._to_float(features.get('ptp')),
                'mean': self._to_float(features.get('mean')),
                'std': self._to_float(features.get('std')),
                'min': self._to_float(features.get('min')),
                'max': self._to_float(features.get('max')),
                'median': self._to_float(features.get('median')),
                'rms': self._to_float(features.get('rms')),
                'noise_level': self._to_float(features.get('noise_level')),
                'freq_1': self._to_float(freq_1),
                'freq_2': self._to_float(freq_2),
                'freq_3': self._to_float(freq_3),
                'kurtosis': self._to_float(features.get('kurtosis')),
                'skewness': self._to_float(features.get('skewness')),
                'data_count': features.get('data_count', 0),
                'is_valid': features.get('is_valid', True),
                'validation_issues': validation_issues,
                'source_zip': features.get('source_zip'),
                'source_bin': features.get('source_bin'),
                'file_size': features.get('file_size'),
                'processed_at': datetime.utcnow()
            }
            
            return record
        
        except Exception as e:
            logger.error(f"Failed to prepare record: {e}")
            return None
    
    
    def _prepare_static_record(self, features: Dict) -> Optional[Dict]:
        """
        정적 센서 특성값을 DB 레코드로 변환
        """
        try:
            if 'sensor_id' not in features or 'hour' not in features:
                return None
            
            record = {
                'time': features['hour'],
                'sensor_id': features['sensor_id'],
                'sensor_type': features.get('sensor_type', 'unknown'),
                'mean': self._to_float(features.get('mean')),
                'ptp': self._to_float(features.get('ptp')),
                'noise_level': self._to_float(features.get('noise_level')),
                'min': self._to_float(features.get('min')),
                'max': self._to_float(features.get('max')),
                'data_count': features.get('data_count', 0),
                'is_valid': features.get('is_valid', True),
                'source_file': features.get('source_file'),
                'processed_at': datetime.utcnow()
            }
            
            return record
        
        except Exception as e:
            logger.error(f"Failed to prepare static record: {e}")
            return None
    
    
    def _to_float(self, value: Any) -> Optional[float]:
        """
        값을 float로 안전하게 변환 (NaN, Inf 처리)
        """
        if value is None:
            return None
        
        try:
            f_value = float(value)
            
            # NaN, Inf 체크
            if not np.isfinite(f_value):
                return None
            
            return f_value
        
        except (ValueError, TypeError):
            return None
    
    
    def check_existing(self, 
                      sensor_ids: List[str],
                      start_time: datetime,
                      end_time: datetime,
                      table: str = 'acceleration') -> DataFrame:
        """
        이미 처리된 데이터 확인
        
        Args:
            sensor_ids: 센서 ID 리스트
            start_time: 시작 시간
            end_time: 종료 시간
            table: 테이블 ('acceleration' or 'static')
        
        Returns:
            기존 데이터 DataFrame (columns: ['sensor_id', 'time'])
        """
        if table == 'acceleration':
            tbl = self.acc_features_table
        else:
            tbl = self.static_features_table
        
        stmt = select(tbl.c.sensor_id, tbl.c.time).where(
            and_(
                tbl.c.sensor_id.in_(sensor_ids),
                tbl.c.time >= start_time,
                tbl.c.time <= end_time
            )
        )
        
        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            df = pd.DataFrame(result.fetchall(), columns=['sensor_id', 'time'])
        
        return df
    
    
    def query_features(self,
                      sensor_id: Optional[str] = None,
                      sensor_ids: Optional[List[str]] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      is_valid: Optional[bool] = None,
                      table: str = 'acceleration',
                      limit: Optional[int] = None) -> DataFrame:
        """
        특성값 조회
        
        Args:
            sensor_id: 단일 센서 ID
            sensor_ids: 센서 ID 리스트
            start_time: 시작 시간
            end_time: 종료 시간
            is_valid: 유효성 필터
            table: 테이블 선택
            limit: 결과 개수 제한
        
        Returns:
            특성값 DataFrame
        """
        if table == 'acceleration':
            tbl = self.acc_features_table
        else:
            tbl = self.static_features_table
        
        # 쿼리 조건 구성
        conditions = []
        
        if sensor_id:
            conditions.append(tbl.c.sensor_id == sensor_id)
        elif sensor_ids:
            conditions.append(tbl.c.sensor_id.in_(sensor_ids))
        
        if start_time:
            conditions.append(tbl.c.time >= start_time)
        
        if end_time:
            conditions.append(tbl.c.time <= end_time)
        
        if is_valid is not None:
            conditions.append(tbl.c.is_valid == is_valid)
        
        # 쿼리 실행
        stmt = select(tbl)
        if conditions:
            stmt = stmt.where(and_(*conditions))
        
        stmt = stmt.order_by(tbl.c.time.desc())
        
        if limit:
            stmt = stmt.limit(limit)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(stmt, conn)
        
        return df
    
    
    def get_statistics(self, 
                      sensor_id: str,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      table: str = 'acceleration') -> Dict:
        """
        센서별 통계 요약
        
        Args:
            sensor_id: 센서 ID
            start_time: 시작 시간
            end_time: 종료 시간
            table: 테이블 선택
        
        Returns:
            통계 딕셔너리
        """
        if table == 'acceleration':
            tbl = self.acc_features_table
            cols = ['ptp', 'noise_level', 'freq_1', 'rms']
        else:
            tbl = self.static_features_table
            cols = ['mean', 'ptp', 'noise_level']
        
        conditions = [tbl.c.sensor_id == sensor_id]
        
        if start_time:
            conditions.append(tbl.c.time >= start_time)
        if end_time:
            conditions.append(tbl.c.time <= end_time)
        
        # 데이터 조회
        df = self.query_features(
            sensor_id=sensor_id,
            start_time=start_time,
            end_time=end_time,
            table=table
        )
        
        if len(df) == 0:
            return {}
        
        # 통계 계산
        stats = {
            'sensor_id': sensor_id,
            'n_records': len(df),
            'time_range': (df['time'].min(), df['time'].max()),
            'valid_rate': df['is_valid'].sum() / len(df) if 'is_valid' in df.columns else None
        }
        
        # 각 컬럼별 통계
        for col in cols:
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    stats[f'{col}_mean'] = float(data.mean())
                    stats[f'{col}_std'] = float(data.std())
                    stats[f'{col}_min'] = float(data.min())
                    stats[f'{col}_max'] = float(data.max())
        
        return stats
    
    
    def get_sensor_list(self, table: str = 'acceleration') -> List[str]:
        """
        등록된 센서 ID 목록 반환
        
        Args:
            table: 테이블 선택
        
        Returns:
            센서 ID 리스트
        """
        if table == 'acceleration':
            tbl = self.acc_features_table
        else:
            tbl = self.static_features_table
        
        stmt = select(tbl.c.sensor_id).distinct()
        
        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            sensor_ids = [row[0] for row in result]
        
        return sorted(sensor_ids)
    
    
    def delete_by_time_range(self,
                            start_time: datetime,
                            end_time: datetime,
                            sensor_id: Optional[str] = None,
                            table: str = 'acceleration') -> int:
        """
        시간 범위로 데이터 삭제 (재처리용)
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            sensor_id: 센서 ID (None이면 전체)
            table: 테이블 선택
        
        Returns:
            삭제된 레코드 수
        """
        if table == 'acceleration':
            tbl = self.acc_features_table
        else:
            tbl = self.static_features_table
        
        conditions = [
            tbl.c.time >= start_time,
            tbl.c.time <= end_time
        ]
        
        if sensor_id:
            conditions.append(tbl.c.sensor_id == sensor_id)
        
        stmt = tbl.delete().where(and_(*conditions))
        
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            deleted_count = result.rowcount
        
        logger.info(f"Deleted {deleted_count} records from {table}")
        return deleted_count
    
    
    def export_to_csv(self,
                     output_path: Path,
                     sensor_id: Optional[str] = None,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     table: str = 'acceleration'):
        """
        데이터를 CSV로 내보내기
        
        Args:
            output_path: 출력 파일 경로
            sensor_id: 센서 ID
            start_time: 시작 시간
            end_time: 종료 시간
            table: 테이블 선택
        """
        df = self.query_features(
            sensor_id=sensor_id,
            start_time=start_time,
            end_time=end_time,
            table=table
        )
        
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} records to {output_path}")
    
    
    def close(self):
        """연결 종료"""
        self.engine.dispose()
        logger.info("Database connection closed")


# 편의 함수들

def create_database_from_config(config: Dict) -> TimeseriesDB:
    """
    설정에서 DB 인스턴스 생성
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        TimeseriesDB 인스턴스
    """
    db_config = config.get('database', {})
    
    return TimeseriesDB(
        host=db_config.get('host', 'localhost'),
        port=db_config.get('port', 5432),
        database=db_config.get('database', 'bridge_monitoring'),
        user=db_config.get('user', 'postgres'),
        password=db_config.get('password', 'postgres'),
        schema=db_config.get('schema', 'public')
    )


if __name__ == "__main__":
    """
    테스트 코드
    """
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("TimescaleDB Connection Test")
    print("="*60)
    
    # DB 연결
    db = TimeseriesDB(
        host='localhost',
        port=5432,
        database='bridge_monitoring',
        user='postgres',
        password='postgres'
    )
    
    try:
        # 1. 테이블 생성
        print("\n[1] Creating tables...")
        db.create_tables(drop_existing=False)
        
        # 2. 테스트 데이터 삽입
        print("\n[2] Inserting test data...")
        test_features = [
            {
                'sensor_id': 'DNA21001',
                'hour': datetime(2024, 1, 1, 0, 0),
                'ptp': 0.0123,
                'mean': 0.0001,
                'std': 0.0023,
                'rms': 0.0025,
                'noise_level': 0.0005,
                'natural_freqs': [2.1, 5.3, 10.2],
                'kurtosis': 3.2,
                'skewness': 0.1,
                'data_count': 360000,
                'is_valid': True,
                'validation_issues': [],
                'source_zip': 'test.zip',
                'source_bin': 'test.bin',
                'file_size': 4320000
            }
        ]
        
        n_inserted = db.insert_acceleration_features(test_features)
        print(f"Inserted {n_inserted} records")
        
        # 3. 데이터 조회
        print("\n[3] Querying data...")
        df = db.query_features(
            sensor_id='DNA21001',
            limit=10
        )
        print(f"Retrieved {len(df)} records")
        if len(df) > 0:
            print(df[['sensor_id', 'time', 'ptp', 'freq_1', 'freq_2', 'freq_3']])
        
        # 4. 센서 목록
        print("\n[4] Getting sensor list...")
        sensors = db.get_sensor_list()
        print(f"Sensors: {sensors[:10]}")  # 처음 10개만
        
        # 5. 통계
        if sensors:
            print(f"\n[5] Getting statistics for {sensors[0]}...")
            stats = db.get_statistics(sensors[0])
            print(stats)
        
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()