#Transform

import json
import pandas as pd
import pyarrow as pa
from pathlib import Path
from typing import Literal, Tuple, Optional
import re
from datetime import datetime


class Transform:
    """
    Transform class for processing YouTube trending data from JSON files.
    
    This class handles the extraction and transformation of trend lists and video details
    from raw JSON data files with timestamp-based naming conventions.
    """
    
    def __init__(self, raw_data_path: Optional[str] = None, 
                 file_prefix: str = "complete_extraction_"):
        """
        Initialize the Transform class.
        
        Args:
            raw_data_path (str): Path to the directory containing raw JSON files
            file_prefix (str): Prefix for the JSON files to process
        """
        self.raw_data_path = Path(raw_data_path) if raw_data_path else Path("../data/raw/extract/")
        self.file_prefix = file_prefix
        self.complete_details = None
        self.trend_ids = None
        self.videos_clean = None
    
    def load_json_file(self, filename: Optional[str] = None) -> dict:
        """
        Load JSON file with the specified naming convention.
        
        Args:
            filename (str, optional): Specific filename to load. If None, finds the latest file.
            
        Returns:
            dict: Loaded JSON data
            
        Raises:
            FileNotFoundError: If no matching files are found
            ValueError: If filename doesn't match expected pattern
        """
        if filename:
            # Validate filename pattern
            pattern = rf"{self.file_prefix}\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}\.json"
            if not re.match(pattern, filename):
                raise ValueError(f"Filename must match pattern: {self.file_prefix}YYYY_MM_DD_HH_MM_SS.json")
            
            file_path = self.raw_data_path / filename
        else:
            # Find the latest file matching the pattern
            pattern = f"{self.file_prefix}*.json"
            resolved_path = self.raw_data_path.resolve()
            matching_files = list(resolved_path.glob(pattern))
            
            if not matching_files:
                raise FileNotFoundError(f"No files found matching pattern: {pattern} in directory: {resolved_path}")
            
            # Sort by filename to get the latest timestamp
            file_path = sorted(matching_files)[-1]
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.complete_details = json.load(f)
        
        return self.complete_details
    
    def extract_trend_ids(self) -> pd.DataFrame:
        """
        Extract and transform trend IDs from the loaded JSON data.
        
        Returns:
            pd.DataFrame: Processed trend IDs with proper data types
        """
        if self.complete_details is None:
            raise ValueError("No data loaded. Call load_json_file() first.")
        
        response = self.complete_details["trend_list"]
        
        # Normalize JSON data
        trend_ids = pd.json_normalize(
            response,
            record_path=['items'],
            meta=['etag', 'nextPageToken', 'prevPageToken', 'kind'],
            meta_prefix='root_',
            errors='ignore'
        )[['id', 'root_kind', 'root_etag', 'root_nextPageToken', 'root_prevPageToken']]
        
        # Rename columns
        trend_ids = trend_ids.rename(columns={'id': 'video_id'})
        
        # Add first appearance date
        trend_ids["first_appearance_date"] = pd.to_datetime('today').strftime('%Y-%m-%d')
        trend_ids["trending_date"] = pd.to_datetime('today').strftime('%Y-%m-%d') 
        
        # Rename columns to final format
        trend_ids.rename(columns={
            'root_kind': 'kind',
            'root_etag': 'etag',
            'root_nextPageToken': 'nextPageToken',
            'root_prevPageToken': 'prevPageToken'
        }, inplace=True)
        
        # Apply data type conversions
        trend_ids = trend_ids.astype({
            'video_id': 'string[pyarrow]',
            'etag': 'string[pyarrow]',
            'nextPageToken': 'string[pyarrow]',
            'prevPageToken': 'string[pyarrow]',
            'kind': 'category'
        })
        
        # Convert date column
        trend_ids["first_appearance_date"] = pd.to_datetime(
            trend_ids["first_appearance_date"], 
            errors='coerce', 
            format='%Y-%m-%d'
        ).astype(pd.ArrowDtype(pa.date32()))
        
        self.trend_ids = trend_ids
        return trend_ids
    
    def extract_video_details(self) -> pd.DataFrame:
        """
        Extract and transform video details from the loaded JSON data.
        
        Returns:
            pd.DataFrame: Processed video details with proper data types
        """
        if self.complete_details is None:
            raise ValueError("No data loaded. Call load_json_file() first.")
        
        # Combine all batches of video details
        all_response_details = []
        for batch in self.complete_details["video_details"]:
            if 'items' in batch:
                all_response_details.extend(batch['items'])
        
        response_details = {'items': all_response_details}


        videos = response_details['items']
        
        # Normalize video data
        videos = pd.json_normalize(videos)
        
        # Rename columns
        videos = videos.rename(columns={
            'snippet.publishedAt': 'published_at',
            'snippet.title': 'title',
            'snippet.channelId': 'channel_id',
            'snippet.thumbnails.default.url': 'thumbnail_url',
            'snippet.channelTitle': 'channel_title',
            'snippet.tags': 'tags',
            'snippet.categoryId': 'category_id',
            'snippet.localized.title': 'localized_title',
            'snippet.localized.description': 'localized_description',
            'contentDetails.duration': 'duration',
            'contentDetails.dimension': 'dimension',
            'contentDetails.definition': 'definition',
            'statistics.viewCount': 'view_count',
            'statistics.likeCount': 'like_count',
            'statistics.dislikeCount': 'dislike_count',
            'statistics.favoriteCount': 'favorite_count',
            'statistics.commentCount': 'comment_count',
            'snippet.defaultAudioLanguage': 'default_language'
        })
        
        # Clean up columns
        videos_clean = videos.drop(columns=[
            col for col in videos.columns 
            if col.startswith(('contentDetails', 'snippet', 'statistics'))
        ]).drop(columns=['dimension', 'tags', 'definition'], errors='ignore')
        
        # Process published_at date
        videos_clean['published_at'] = pd.to_datetime(
            videos_clean['published_at'], 
            errors='coerce'
        ).dt.date
        videos_clean['published_at'] = videos_clean['published_at'].astype(
            pd.ArrowDtype(pa.date32())
        )
        
        # Process duration
        videos_clean['duration'] = pd.to_timedelta(
            videos_clean['duration'], 
            errors='coerce'
        ).dt.total_seconds().astype('float32[pyarrow]')
        
        # Apply data type conversions
        videos_clean = videos_clean.astype({
            'kind': 'category',
            'etag': 'string[pyarrow]',
            'id': 'string[pyarrow]',
            'channel_id': 'string[pyarrow]',
            'thumbnail_url': 'string[pyarrow]',
            'channel_title': 'string[pyarrow]',
            'category_id': 'category',
            'localized_title': 'string[pyarrow]',
            'localized_description': 'string[pyarrow]',
            'default_language': 'category',
            'view_count': 'Int64[pyarrow]',
            'like_count': 'Int64[pyarrow]',
            'comment_count': 'Int32[pyarrow]',
        })
        
        # Handle favorite_count with sparse dtype
        videos_clean["favorite_count"] = videos_clean["favorite_count"].astype(
            pd.SparseDtype("int32", 0)
        )
        
        self.videos_clean = videos_clean
        return videos_clean
    
    def process_all(self, filename: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process both trend IDs and video details in one call.
        
        Args:
            filename (str, optional): Specific filename to process. If None, processes latest file.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (trends, videos)
        """
        self.load_json_file(filename)
        trend_ids = self.extract_trend_ids()
        videos_clean = self.extract_video_details()

        self.trend_ids = trend_ids.merge(
            videos_clean[['id', 'view_count']],
            left_on='video_id',
            right_on='id',
            how='left'
        ).drop(columns=['id'])

        return self.trend_ids.copy(), videos_clean
    
    def save_processed_data(self, output_path: str | Path = "../data/processed/", 
                          file_suffix: Optional[str] = None,
                          format : Literal['parquet', 'csv'] = 'parquet') -> None:
        """
        Save processed DataFrames to parquet files.
        
        Args:
            output_path (str): Path to save processed files
            file_suffix (str, optional): Suffix for output files. If None, uses timestamp.
            format (Literal['parquet', 'csv']): Format to save files, either 'parquet' or 'csv'.
        """
        if self.trend_ids is None or self.videos_clean is None:
            raise ValueError("No processed data available. Call process_all() first.")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if file_suffix is None:
            file_suffix = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        # Save trend IDs
        if format==None:
            format = 'parquet'
            
        trend_ids_path = output_path / f"trend_ids_{file_suffix}.{format}"

        temp_trend_ids = self.trend_ids.copy()

        # Save video details
        videos_path = output_path / f"video_details_{file_suffix}.{format}"

        # Convert sparse column to dense before saving
        temp_videos_clean = self.videos_clean.copy()
        temp_videos_clean["favorite_count"] = temp_videos_clean["favorite_count"].sparse.to_dense()

        if format == 'csv':
            temp_videos_clean.to_csv(videos_path.with_suffix('.csv'), index=False)
            temp_trend_ids.to_csv(trend_ids_path.with_suffix('.csv'), index=False)
        else:
            temp_videos_clean.to_parquet(videos_path, index=False)
            temp_trend_ids.to_parquet(trend_ids_path, index=False)

        print(f"Saved processed data to:")
        print(f"  - {trend_ids_path}")
        print(f"  - {videos_path}")


# Example usage
if __name__ == "__main__":
    # Initialize transformer
    transformer = Transform()
    
    # Process specific file
    trend_ids, videos = transformer.process_all()
    
    # Display info about processed data
    print("Trend IDs shape:", trend_ids.shape)
    print("Videos shape:", videos.shape)
    print("\nTrend IDs columns:", list(trend_ids.columns))
    print("Videos columns:", list(videos.columns))
    
    # Save processed data
    transformer.save_processed_data(format="csv")
    transformer.save_processed_data(format="parquet")