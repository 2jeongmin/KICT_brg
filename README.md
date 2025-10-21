# 교량 센서 데이터 특성값 추출 시스템

한국건설기술연구원 장기 계측 데이터(10TB, 100개 교량, 3년)로부터 시간 단위 특성값을 추출하고 TimescaleDB에 저장하는 시스템

## 📋 프로젝트 개요

- **목적**: 교량 가속도/정적 센서 데이터의 시간 단위 요약 특성값 추출
- **데이터 규모**: 10TB (가속도 센서 500채널, 100Hz, 3년)
- **추출 특성값**: Peak-to-Peak, 노이즈레벨, 고유진동수(FFT), 통계값
- **저장소**: PostgreSQL + TimescaleDB
- **처리 방식**: Zip 스트리밍 (디스크 용량 증가 0)

## 🚀 빠른 시작

### 1. 설치
```bash
# Python 3.8+ 필요
pip install -r requirements.txt
```

### 2. 설정 파일 작성

`config/default.yaml` 생성:
```yaml
data:
  root_dir: "Z:/05. Data/01. Under_Process/027. KICT_BMAPS/upload"
  mode: "stream"  # stream | extract_delete | temp_dir
  
features:
  extract: ["ptp", "noise", "fft", "statistics"]
  sampling_rate: 100  # Hz
  fft_freq_range: [0.5, 20]  # Hz
  
database:
  host: "localhost"
  port: 5432
  database: "bridge_monitoring"
  user: "postgres"
  password: "your_password"
  
processing:
  batch_size: 100
  n_workers: 4
  max_files: null  # null = 전체 처리
```

### 3. 실행
```bash
# 기본 실행
python main.py --config config/default.yaml

# 테스트 (10개 파일만)
python main.py --config config/test.yaml --max-files 10

# 특정 센서만
python main.py --config config/default.yaml --sensors DNA21001,DNA21002
```

## 📊 추출 특성값

### 가속도 센서 (100Hz, 시간당)

| 특성값 | 설명 | 단위 |
|--------|------|------|
| `ptp` | Peak-to-Peak (최대-최소) | m/s² |
| `mean` | 평균 | m/s² |
| `std` | 표준편차 | m/s² |
| `rms` | RMS | m/s² |
| `noise_level` | 노이즈 레벨 (정적 구간 std) | m/s² |
| `natural_freqs` | 고유진동수 (FFT 기반, 최대 3개) | Hz |
| `kurtosis` | 첨도 | - |
| `skewness` | 왜도 | - |

### 정적 센서 (분 단위 → 시간 요약)

| 특성값 | 설명 |
|--------|------|
| `mean` | 시간 평균 |
| `ptp` | Peak-to-Peak |
| `noise_level` | 표준편차 |

## 🗄️ 데이터베이스 스키마

### 가속도 특성값 테이블
```sql
CREATE TABLE acc_hourly_features (
    time TIMESTAMPTZ NOT NULL,
    sensor_id VARCHAR(50) NOT NULL,
    ptp FLOAT,
    mean FLOAT,
    std FLOAT,
    rms FLOAT,
    noise_level FLOAT,
    natural_freqs FLOAT[],  -- 최대 3개
    kurtosis FLOAT,
    skewness FLOAT,
    data_count INTEGER,
    is_valid BOOLEAN,
    source_file VARCHAR(255),
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (time, sensor_id)
);

SELECT create_hypertable('acc_hourly_features', 'time');
CREATE INDEX ON acc_hourly_features (sensor_id, time DESC);
```

## 📁 프로젝트 구조
```
KICT_brg/
├── main.py                    # 메인 실행 파일
├── requirements.txt
├── README.md
│
├── config/                    # 설정 파일
│   ├── default.yaml
│   └── test.yaml
│
├── src/                       # 핵심 모듈
│   ├── io_utils.py           # Binary I/O
│   ├── zip_handler.py        # Zip 스트리밍
│   ├── feature_extraction.py # 특성값 추출
│   ├── database.py           # DB 연동
│   └── pipeline.py           # 전체 파이프라인
│
├── scripts/                   # 유틸리티 스크립트
│   ├── check_data.py
│   └── export_results.py
│
├── tests/                     # 테스트
└── logs/                      # 로그 파일
```

## 🔧 주요 기능

### 1. Zip 스트리밍 처리
- **디스크 용량 증가 0**: 압축 해제 없이 메모리에서 직접 처리
- **Dropbox 안전**: 클라우드 동기화 환경에서도 문제 없음

### 2. 병렬 처리
- multiprocessing 기반 고속 처리
- 권장: CPU 코어 수만큼 워커 설정

### 3. 데이터 검증
- 시간 간격 이상 감지 (100Hz 기준)
- 값 범위 체크 (가속도: -100 ~ 100 m/s²)
- 데이터 개수 검증 (시간당 ~360,000개)

### 4. 에러 핸들링
- 손상된 파일 자동 건너뛰기
- 상세한 에러 로깅
- 중단 후 재시작 가능

## 📈 성능

### 처리 속도
- 파일당 처리 시간: ~0.3초 (스트리밍 모드)
- 예상 전체 소요 시간: ~4-6시간 (50,000개 파일 기준, 4 workers)

### 메모리 사용량
- 파일당: ~3MB
- 배치 100개: ~300MB
- 권장 시스템 메모리: 16GB 이상

### 디스크 사용량
- 스트리밍 모드: 증가 없음
- DB 저장 공간: ~100GB 예상

## 🛠️ 개발 가이드

### 새로운 특성값 추가
```python
# src/feature_extraction.py

class FeatureExtractor:
    def extract_my_feature(self, df: DataFrame) -> float:
        """새로운 특성값 추출"""
        values = df['value'].values
        # 계산 로직
        return result
    
    def extract_all(self, df: DataFrame) -> Dict:
        features = self._extract_basic(df)
        features['my_feature'] = self.extract_my_feature(df)
        return features
```

### 커스텀 처리 파이프라인
```python
from src.pipeline import ProcessingPipeline

pipeline = ProcessingPipeline(config)

# 단계별 실행
pipeline.step1_load_files()
pipeline.step2_extract_features()
pipeline.step3_save_to_db()
```

## 📝 로그

로그는 `logs/processing_YYYYMMDD_HHMMSS.log`에 저장됩니다.
```bash
# 로그 확인
tail -f logs/processing_20241028_143022.log

# 에러만 확인
grep ERROR logs/processing_*.log
```

## 🔍 데이터 조회 예시
```sql
-- 특정 센서의 최근 24시간 데이터
SELECT time, ptp, natural_freqs
FROM acc_hourly_features
WHERE sensor_id = 'DNA21001'
  AND time > NOW() - INTERVAL '24 hours'
ORDER BY time DESC;

-- 고유진동수 변화 추이
SELECT sensor_id, 
       time_bucket('1 day', time) AS day,
       AVG(natural_freqs[1]) AS avg_freq
FROM acc_hourly_features
WHERE sensor_id = 'DNA21001'
GROUP BY sensor_id, day
ORDER BY day;
```"# KICT_brg" 
