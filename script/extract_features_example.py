#!/usr/bin/env python3
# scripts/extract_features_example.py
"""
특성값 추출 전체 워크플로우 예시
zip 파일 → 특성값 추출 → 결과 저장
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.zip_handler import ZipHandler
from src.feature_extraction import BatchFeatureExtractor, FeatureExtractor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    전체 워크플로우 실행
    """
    # 설정
    root_dir = Path("Z:/05. Data/01. Under_Process/027. KICT_BMAPS/upload")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 테스트 모드
    MAX_FILES = 10
    
    logger.info("Starting feature extraction pipeline")
    logger.info(f"Input directory: {root_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max files: {MAX_FILES}")
    
    # 1. Zip 핸들러 초기화
    zip_handler = ZipHandler(mode='stream')
    
    # 2. 특성값 추출기 초기화
    feature_extractor = FeatureExtractor(
        sampling_rate=100,
        fft_freq_range=(0.5, 20),
        fft_n_peaks=3,
        noise_method='static_std'
    )
    batch_extractor = BatchFeatureExtractor(feature_extractor)
    
    # 3. 배치 처리
    all_features = []
    batch_count = 0
    
    for zip_results in zip_handler.process_all_zips(root_dir, max_files=MAX_FILES):
        batch_count += 1
        logger.info(f"Processing batch {batch_count}: {len(zip_results)} files")
        
        # 특성값 추출
        features_list = batch_extractor.extract_from_zip_results(
            zip_results, 
            sensor_type='acceleration'
        )
        
        all_features.extend(features_list)
        
        # 중간 저장 (10개 배치마다)
        if batch_count % 10 == 0:
            features_df = batch_extractor.to_dataframe(all_features)
            temp_file = output_dir / f"features_temp_batch{batch_count}.csv"
            features_df.to_csv(temp_file, index=False)
            logger.info(f"Saved intermediate results to {temp_file}")
    
    # 4. 최종 결과 저장
    if all_features:
        features_df = batch_extractor.to_dataframe(all_features)
        
        # CSV 저장
        csv_file = output_dir / "features_all.csv"
        features_df.to_csv(csv_file, index=False)
        logger.info(f"Saved final results to {csv_file}")
        
        # Excel 저장 (작은 경우만)
        if len(features_df) < 10000:
            excel_file = output_dir / "features_all.xlsx"
            features_df.to_excel(excel_file, index=False)
            logger.info(f"Saved Excel to {excel_file}")
        
        # Pickle 저장 (빠른 로드용)
        pkl_file = output_dir / "features_all.pkl"
        features_df.to_pickle(pkl_file)
        logger.info(f"Saved pickle to {pkl_file}")
    
    # 5. 통계 출력
    stats = zip_handler.get_statistics()
    logger.info("="*60)
    logger.info("Processing Statistics")
    logger.info("="*60)
    logger.info(f"Zip files processed: {stats['processed_zips']}/{stats['total_zips']}")
    logger.info(f"Bin files processed: {stats['processed_bins']}/{stats['total_bins']}")
    logger.info(f"Success rate: {stats['zip_success_rate']*100:.1f}%")
    
    if all_features:
        logger.info(f"Total features extracted: {len(all_features)}")
        logger.info("\nFeature Summary:")
        logger.info(features_df[['ptp', 'noise_level', 'freq_1']].describe())
    
    logger.info("="*60)
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()