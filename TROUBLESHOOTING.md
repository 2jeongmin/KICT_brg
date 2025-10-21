# 트러블슈팅 가이드

## 목차
1. [설치 문제](#설치-문제)
2. [실행 문제](#실행-문제)
3. [성능 문제](#성능-문제)
4. [데이터 문제](#데이터-문제)
5. [DB 문제](#db-문제)

---

## 설치 문제

### Python 패키지 설치 실패

#### 문제: `psycopg2` 설치 오류
```
Error: pg_config executable not found
```

**해결:**
```bash
# Ubuntu/Debian
sudo apt-get install libpq-dev python3-dev

# CentOS/RHEL
sudo yum install postgresql-devel python3-devel

# Windows
# psycopg2-binary 사용 (requirements.txt에 이미 포함)
pip install psycopg2-binary
```

#### 문제: `scipy` 설치 오류

**해결:**
```bash
# 먼저 numpy 설치
pip install numpy

# 그 다음 scipy
pip install scipy

# 또는 Anaconda 사용
conda install scipy
```

---

## 실행 문제

### 문제: "Config file not found"

**원인:** 설정 파일 경로 오류

**해결:**
```bash
# 현재 디렉토리 확인
pwd

# 설정 파일 확인
ls config/

# 절대 경로 사용
python main.py --config /full/path/to/config/default.yaml
```

### 문제: "Permission denied"

**원인:** 파일 접근 권한 부족

**해결:**
```bash
# Linux/macOS
chmod +x scripts/*.sh
chmod +x scripts/*.py

# Windows
# 관리자 권한으로 실행
```

### 문제: 모듈을 찾을 수 없음
```
ModuleNotFoundError: No module named 'src'
```

**해결:**
```bash
# PYTHONPATH 설정
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 또는 main.py에서 자동 처리 (이미 구현됨)
```

---

## 성능 문제

### 문제: 처리 속도 0.5 files/s 미만

**진단:**
```bash
# CPU 사용률 확인
top

# 디스크 I/O 확인
iostat -x 1

# 네트워크 확인 (네트워크 드라이브 사용 시)
iftop
```

**해결 1: 병렬 처리 증가**
```yaml
processing:
  n_workers: 8  # CPU 코어 수만큼
```

**해결 2: 로컬 복사**
```bash
# 네트워크 드라이브 데이터를 로컬로 복사
rsync -av --progress /network/data/ /local/data/
```

**해결 3: SSD 사용**
- HDD → SSD로 변경 시 3-5배 속도 향상

### 문제: 메모리 사용량 과다

**진단:**
```bash
# 메모리 사용량 확인
free -h

# 프로세스별 메모리
ps aux --sort=-%mem | head
```

**해결:**
```yaml
processing:
  batch_size: 50   # 100 → 50
  n_workers: 2     # 4 → 2
```

---

## 데이터 문제

### 문제: "Empty DataFrame" 다수 발생

**원인:** 
1. Zip 파일 손상
2. Bin 파일 형식 불일치
3. 파일 크기 0

**진단:**
```bash
# 손상된 파일 찾기
find /data -name "*.zip" -size 0

# Zip 파일 무결성 검사
unzip -t file.zip
```

**해결:**
```bash
# 손상된 파일 제거
find /data -name "*.zip" -size 0 -delete

# 재처리
python main.py --config config/default.yaml
```

### 문제: 특성값이 비정상적

**예:** PTP가 항상 0 또는 매우 큼

**진단:**
```python
# 샘플 데이터 확인
from src.io_utils import get_df_from_bin

df = get_df_from_bin('sample.bin')
print(df.describe())
print(df['value'].hist())
```

**해결:**
1. 샘플링 주파수 확인
```yaml
features:
  acceleration:
    sampling_rate: 100  # 정확한 값 확인
```

2. 단위 확인 (m/s² vs. g)

3. 센서 캘리브레이션 필요 시 스케일 조정
```python
# feature_extraction.py에서
df['value'] = df['value'] * scale_factor
```

---

## DB 문제

### 문제: "Connection refused"

**진단:**
```bash
# PostgreSQL 실행 여부
sudo systemctl status postgresql

# 포트 열려있는지 확인
sudo netstat -tlnp | grep 5432
```

**해결:**
```bash
# PostgreSQL 시작
sudo systemctl start postgresql

# 자동 시작 설정
sudo systemctl enable postgresql
```

### 문제: "Too many connections"

**원인:** DB 연결 수 초과

**해결:**
```sql
-- 현재 연결 수 확인
SELECT count(*) FROM pg_stat_activity;

-- 최대 연결 수 확인
SHOW max_connections;

-- 늘리기 (postgresql.conf)
max_connections = 200

-- PostgreSQL 재시작
sudo systemctl restart postgresql
```

### 문제: 느린 쿼리

**진단:**
```sql
-- 느린 쿼리 찾기
SELECT query, mean_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

**해결:**
```sql
-- 인덱스 생성
CREATE INDEX idx_time ON acc_hourly_features (time DESC);
CREATE INDEX idx_sensor ON acc_hourly_features (sensor_id);
-- VACUUM 실행
VACUUM ANALYZE acc_hourly_features;
-- TimescaleDB 압축 활성화
ALTER TABLE acc_hourly_features SET (
timescaledb.compress,
timescaledb.compress_segmentby = 'sensor_id'
);
SELECT add_compression_policy('acc_hourly_features', INTERVAL '7 days');

### 문제: 디스크 공간 부족

**진단:**
```bash
# 디스크 사용량
df -h

# DB 크기 확인
psql -U postgres -d bridge_monitoring -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"
```

**해결:**
```sql
-- 오래된 데이터 삭제
DELETE FROM acc_hourly_features 
WHERE time < NOW() - INTERVAL '2 years';

-- VACUUM FULL (테이블 잠금 주의)
VACUUM FULL acc_hourly_features;

-- TimescaleDB chunk 삭제
SELECT drop_chunks('acc_hourly_features', INTERVAL '1 year');
```

---

## 특정 에러 메시지 해결

### 에러: "BadZipFile: File is not a zip file"

**원인:** 손상되었거나 불완전한 zip 파일

**해결:**
```bash
# 해당 파일 확인
file suspicious_file.zip

# 손상된 파일 제거
rm suspicious_file.zip

# 로그에서 손상된 파일 목록 추출
grep "BadZipFile" logs/*.log | cut -d: -f2 > corrupted_files.txt
```

### 에러: "KeyError: 'natural_freqs'"

**원인:** 특성값 추출 실패

**해결:**
1. 설정 파일 확인:
```yaml
features:
  acceleration:
    extract: ["ptp", "noise", "fft"]  # 'fft' 포함 확인
```

2. 데이터 길이 확인 (FFT는 최소 1000개 샘플 필요)

### 에러: "sqlalchemy.exc.DataError: invalid input syntax"

**원인:** 잘못된 데이터 타입 또는 NaN 값

**해결:**
```python
# database.py의 _to_float() 함수 확인 (이미 구현됨)
# NaN, Inf 값은 자동으로 None으로 변환
```

### 에러: "Checkpoint is too old"

**원인:** 1시간 이상 지난 체크포인트

**해결:**
```bash
# 체크포인트 무시하고 새로 시작
python scripts/manage_checkpoint.py clear

# 또는 수동 삭제
rm checkpoints/pipeline_checkpoint.json
```

---

## 긴급 복구 절차

### 시나리오 1: 처리 중 시스템 다운

**복구 단계:**

1. 시스템 재부팅 후 PostgreSQL 상태 확인
```bash
sudo systemctl status postgresql
```

2. DB 무결성 검사
```sql
psql -U postgres -d bridge_monitoring
\dt
SELECT COUNT(*) FROM acc_hourly_features;
```

3. 체크포인트 확인
```bash
python scripts/manage_checkpoint.py show
```

4. 재시작
```bash
python main.py --config config/default.yaml
```

### 시나리오 2: DB 손상

**복구 단계:**

1. 백업 확인
```bash
ls -lh /backup/bridge_monitoring_*.dump
```

2. DB 재생성
```bash
dropdb -U postgres bridge_monitoring
createdb -U postgres bridge_monitoring
```

3. 백업 복원
```bash
pg_restore -U postgres -d bridge_monitoring /backup/latest.dump
```

4. 테이블 재생성 (백업 없는 경우)
```bash
python scripts/init_database.py --config config/default.yaml --drop
```

### 시나리오 3: 대량의 잘못된 데이터

**복구 단계:**

1. 문제 데이터 식별
```sql
SELECT sensor_id, COUNT(*)
FROM acc_hourly_features
WHERE is_valid = false
GROUP BY sensor_id;
```

2. 해당 데이터 삭제
```sql
DELETE FROM acc_hourly_features 
WHERE is_valid = false;
```

3. 재처리
```bash
python main.py --config config/default.yaml --reprocess
```

---

## 디버깅 팁

### 로그 레벨 높이기
```yaml
# config/debug.yaml
logging:
  level: "DEBUG"  # INFO → DEBUG
```
```bash
python main.py --config config/debug.yaml --max-files 1
```

### 단일 파일 테스트
```python
# test_single_file.py
from pathlib import Path
from src.zip_handler import ZipHandler
from src.feature_extraction import FeatureExtractor, BatchFeatureExtractor

# 단일 파일 처리
zip_file = Path("path/to/test.zip")

handler = ZipHandler(mode='stream')
results = handler.process_single_zip(zip_file)

print(f"Processed {len(results)} bin files")
for r in results:
    print(f"  {r['sensor_id']}: {r['n_records']} records")
    print(f"  DataFrame shape: {r['df'].shape}")
```

### Python Profiler 사용
```python
# profiling.py
import cProfile
import pstats
from src.pipeline import ProcessingPipeline

config = {...}  # 설정 로드

profiler = cProfile.Profile()
profiler.enable()

pipeline = ProcessingPipeline(config)
results = pipeline.run()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 상위 20개
```

### 메모리 프로파일링
```bash
# memory_profiler 설치
pip install memory_profiler

# 사용
python -m memory_profiler main.py --config config/test.yaml --max-files 5
```

---

## 문의 전 체크리스트

문제 발생 시 아래 정보를 수집하여 문의:

**시스템 환경:**
```bash
# OS 정보
uname -a  # Linux
systeminfo  # Windows

# Python 버전
python --version

# 설치된 패키지
pip list

# 디스크 공간
df -h

# 메모리
free -h
```

**설정 정보:**
```bash
# 설정 파일 내용
cat config/default.yaml
```

**에러 로그:**
```bash
# 최근 에러 로그
tail -n 50 logs/processing_*.log
```

**DB 상태:**
```sql
-- 테이블 크기
SELECT pg_size_pretty(pg_total_relation_size('acc_hourly_features'));

-- 레코드 수
SELECT COUNT(*) FROM acc_hourly_features;

-- 연결 수
SELECT count(*) FROM pg_stat_activity;
```

---

## 추가 리소스

### 공식 문서
- PostgreSQL: https://www.postgresql.org/docs/
- TimescaleDB: https://docs.timescale.com/
- Python pandas: https://pandas.pydata.org/docs/
- scipy: https://docs.scipy.org/