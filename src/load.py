#Load

# load.py

import pandas as pd
import psycopg as pg
from pathlib import Path
from typing import Optional, Literal, List, Dict, Union, Tuple
from datetime import datetime
import logging
from abc import ABC, abstractmethod
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor(ABC):
    """Abstract base class for data preprocessing steps."""
    
    @abstractmethod
    def process(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Process the DataFrame and return the modified version."""
        pass


class BasicCleanupPreprocessor(DataPreprocessor):
    """Basic data cleanup preprocessor."""
    
    def process(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Basic cleanup: remove duplicates, handle nulls."""
        logger.info(f"Applying basic cleanup to {table_name}")
        
        # Remove duplicates
        original_count = len(df)
        df = df.drop_duplicates()
        if len(df) < original_count:
            logger.info(f"Removed {original_count - len(df)} duplicate rows from {table_name}")
        
        # Log null counts
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.info(f"Null counts in {table_name}: {null_counts[null_counts > 0].to_dict()}")
        
        return df


class YouTubeDataLoader:
    """
    Load processed YouTube data into PostgreSQL using psycopg3.
    
    Handles both CSV and Parquet files, with automatic file detection
    and modular preprocessing pipeline.
    """
    
    def __init__(self, 
                 processed_data_path: str = "../data/processed/",
                 db_config: Optional[Dict[str, str]] = None,
                 preprocessors: Optional[List[DataPreprocessor]] = None):
        """
        Initialize the loader.
        
        Args:
            processed_data_path: Path to processed data directory
            db_config: Database connection configuration
            preprocessors: List of preprocessor classes to apply before loading
        """
        self.processed_data_path = Path(processed_data_path)
        self.db_config = db_config or self._default_db_config()
        self.preprocessors = preprocessors or [BasicCleanupPreprocessor()]
        
        self.trend_ids_df = None
        self.videos_df = None
        
    def _default_db_config(self) -> Dict[str, str]:
        """Default database configuration for local PostgreSQL."""
        return {
            'host': 'localhost',
            'port': '5432',
            'dbname': 'youtube_analytics',
            'user': 'postgres',
            'password': 'password'
        }
    
    def find_latest_files(self, 
                         file_type: Optional[Literal['trend_ids', 'video_details', 'both']] = None,
                         format_preference: Literal['parquet', 'csv'] = 'parquet') -> Dict[str, Optional[Path]]:
        """
        Find the latest processed files in the directory.
        
        Args:
            file_type: Type of files to find ('trend_ids', 'video_details', or 'both')
            format_preference: Preferred format when both parquet and csv exist
            
        Returns:
            Dictionary with file paths for each data type
        """

        if file_type is None:
            file_type = 'both'

        if not self.processed_data_path.exists():
            raise FileNotFoundError(f"Processed data directory not found: {self.processed_data_path}")

        files: Dict[str, Path | None] = {'trend_ids': None, 'video_details': None}

        # Define patterns for each file type
        patterns = {
            'trend_ids': r'trend_ids_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.(csv|parquet)',
            'video_details': r'video_details_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.(csv|parquet)'
        }
        
        # Determine which file types to search for
        search_types = ['trend_ids', 'video_details'] if file_type == 'both' else [file_type]
        
        for data_type in search_types:
            if data_type not in patterns:
                continue
                
            pattern = patterns[data_type]
            matching_files = []
            
            # Find all matching files
            for file_path in self.processed_data_path.iterdir():
                if file_path.is_file():
                    match = re.match(pattern, file_path.name)
                    if match:
                        timestamp = match.group(1)
                        file_format = match.group(2)
                        matching_files.append((timestamp, file_format, file_path))
            
            if not matching_files:
                logger.warning(f"No {data_type} files found")
                continue
            
            # Sort by timestamp (most recent first)
            matching_files.sort(key=lambda x: x[0], reverse=True)
            
            # Group by timestamp to handle cases where both formats exist
            timestamps = {}
            for timestamp, file_format, file_path in matching_files:
                if timestamp not in timestamps:
                    timestamps[timestamp] = {}
                timestamps[timestamp][file_format] = file_path
            
            # Select the most recent timestamp
            latest_timestamp = list(timestamps.keys())[0]
            available_formats = timestamps[latest_timestamp]
            
            # Choose format based on preference and availability
            if format_preference in available_formats:
                selected_file = available_formats[format_preference]
            else:
                # Fallback to any available format
                selected_file = list(available_formats.values())[0]
            
            files[data_type] = selected_file
            logger.info(f"Selected {data_type} file: {selected_file.name}")
        
        return files
    
    def load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single file (CSV or Parquet).
        
        Args:
            file_path: Path to the file
            
        Returns:
            Loaded DataFrame
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading file: {file_path.name}")
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def load_data(self, 
                  trend_ids_file: Optional[Union[str, Path]] = None,
                  videos_file: Optional[Union[str, Path]] = None,
                  file_type: Optional[Literal['trend_ids', 'video_details', 'both']] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load trend IDs and/or video details data.
        
        Args:
            trend_ids_file: Specific trend IDs file to load
            videos_file: Specific video details file to load  
            file_type: Type of files to auto-detect if specific files not provided
            
        Returns:
            Tuple of (trend_ids_df, videos_df)
        """
        if file_type is None:
            file_type = 'both'

        trend_ids_df, videos_df = None, None
        
        # If specific files provided, use them
        if trend_ids_file:
            trend_ids_df = self.load_file(Path(trend_ids_file))
        if videos_file:
            videos_df = self.load_file(Path(videos_file))
        
        # Auto-detect files if not provided
        if trend_ids_file is None or videos_file is None:
            latest_files = self.find_latest_files(file_type=file_type)
            
            if trend_ids_file is None and latest_files['trend_ids']:
                trend_ids_df = self.load_file(latest_files['trend_ids'])
            
            if videos_file is None and latest_files['video_details']:
                videos_df = self.load_file(latest_files['video_details'])
        
        # Store in instance variables
        self.trend_ids_df = trend_ids_df
        self.videos_df = videos_df
        
        return trend_ids_df, videos_df
    
    def apply_preprocessors(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        Apply all preprocessing steps to the DataFrame.
        
        Args:
            df: DataFrame to preprocess
            table_name: Name of the target table
            
        Returns:
            Preprocessed DataFrame
        """
        for preprocessor in self.preprocessors:
            df = preprocessor.process(df, table_name)
        
        return df
    
    def create_tables(self, conn: pg.Connection) -> None:
        """
        Create database tables if they don't exist.
        
        Args:
            conn: Database connection
        """
        logger.info("Creating database tables")
        

        # Create videos table
        videos_sql = """
        CREATE TABLE IF NOT EXISTS videos (
            id VARCHAR(50) PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            published_at DATE,
            channel_id VARCHAR(100),
            channel_title VARCHAR(50),
            category_id SMALLINT,
            duration INTEGER,
            view_count BIGINT,
            like_count BIGINT,
            comment_count BIGINT
        );
        """

        # Create trending_events table
        trending_events_sql = """
        CREATE TABLE IF NOT EXISTS trending_events (
            video_id VARCHAR(50),
            trending_date DATE,
            first_appearance_date DATE,
            view_count BIGINT,
            PRIMARY KEY (video_id, trending_date),
            FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
        );
        """

        
        with conn.cursor() as cur:
            cur.execute(videos_sql)
            cur.execute(trending_events_sql)

            # Create indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trend_ids_first_appearance ON trending_events(first_appearance_date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_video_details_published_at ON videos(published_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_video_details_channel_id ON videos(channel_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_video_details_category_id ON videos(category_id);")

        conn.commit()
        logger.info("Database tables created successfully")

    def upsert_data(self, conn: pg.Connection, df: pd.DataFrame, table_name: Literal['trending_events', 'videos']) -> None:
        """
        Insert or update data in the database table.
        
        Args:
            conn: Database connection
            df: DataFrame to insert
            table_name: Target table name [trending_events, videos]
        """
        if df is None or df.empty:
            logger.warning(f"No data to insert for table: {table_name}")
            return
        logger.info(f"Upserting {len(df)} rows to {table_name}")

        # Expected columns for each table
        table_columns = {
            'trending_events': ["video_id", "trending_date", "first_appearance_date", "view_count"],
            'videos': ["id", "title", "published_at", "channel_id", "channel_title", 
                  "category_id", "duration", "view_count", "like_count", "comment_count"]
        }

        # Prepare DataFrame
        df = df.copy()
        
        # Add trending_date for trending_events table
        if table_name == 'trending_events':
            df['trending_date'] = datetime.now()
        
        # Select only required columns
        required_cols = table_columns[table_name]
        df = df[required_cols]

        # Clean null values - convert pandas nulls to Python None
        df_clean = df.copy()
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].where(pd.notnull(df_clean[col]), None)
        
        # Convert to list of tuples for insertion
        values = [tuple(None if pd.isna(val) or val is pd.NaT else val for val in row) 
             for row in df_clean.values]

        # Prepare SQL components
        columns = list(df.columns)
        columns_str = ', '.join(columns)
        placeholders = ', '.join(['%s'] * len(columns))
        
        # Define conflict resolution
        conflict_targets = {
            'videos': "(id)",
            'trending_events': "(video_id, trending_date)"
        }
        
        # Exclude primary key columns from updates
        excluded_from_update = {'videos': ['id'], 'trending_events': ['video_id', 'trending_date']}
        update_columns = [col for col in columns if col not in excluded_from_update[table_name]]
        update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_columns])

        # Log any missing columns
        missing_cols = set(required_cols) - set(columns)
        if missing_cols:
            logger.warning(f"Missing columns for {table_name}: {missing_cols}")

        # Execute upsert
        upsert_sql = f"""
        INSERT INTO {table_name} ({columns_str})
        VALUES ({placeholders})
        ON CONFLICT {conflict_targets[table_name]} DO UPDATE SET {update_clause}
        """
        
        with conn.cursor() as cur:
            cur.executemany(upsert_sql, values) # type: ignore

        conn.commit()
        logger.info(f"Successfully upserted {len(df)} rows to {table_name}")
    
    def load_to_database(self, 
                        trend_ids_file: Optional[Union[str, Path]] = None,
                        videos_file: Optional[Union[str, Path]] = None,
                        file_type: Optional[Literal['trend_ids', 'video_details', 'both']] = None,
                        create_tables: bool = True) -> None:
        """
        Complete pipeline: load data, preprocess, and insert to database.
        
        Args:
            trend_ids_file: Specific trend IDs file to load
            videos_file: Specific video details file to load
            file_type: Type of files to auto-detect if specific files not provided
            create_tables: Whether to create tables if they don't exist
        """

        if file_type is None:
            file_type = 'both'

        logger.info("Starting data loading pipeline")
        
        # Load data
        trend_ids_df, videos_df = self.load_data(trend_ids_file, videos_file, file_type)
        
        # Connect to database
        try:
            with pg.connect(**self.db_config) as conn: # type: ignore
                logger.info("Connected to database")
                
                # Create tables if requested
                if create_tables:
                    self.create_tables(conn)
                
                # Process and load video_details first to satisfy foreign key constraints
                if videos_df is not None:
                    processed_videos = self.apply_preprocessors(videos_df, 'video_details')
                    self.upsert_data(conn, processed_videos, 'videos')

                # Process and load trend_ids after videos
                if trend_ids_df is not None:
                    processed_trend_ids = self.apply_preprocessors(trend_ids_df, 'trend_ids')
                    self.upsert_data(conn, processed_trend_ids, 'trending_events')
                
                logger.info("Data loading pipeline completed successfully")
                
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise


# Custom preprocessor examples
class ViewCountFilterPreprocessor(DataPreprocessor):
    """Filter records based on view count threshold."""
    
    def __init__(self, min_view_count: int = 1000):
        self.min_view_count = min_view_count
    
    def process(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        if 'view_count' in df.columns:
            original_count = len(df)
            df = df[df['view_count'] >= self.min_view_count]
            logger.info(f"Filtered {original_count - len(df)} rows with view_count < {self.min_view_count}")
        return df


class CategoryMappingPreprocessor(DataPreprocessor):
    """Map category IDs to category names."""
    
    def __init__(self):
        # YouTube category ID mappings (subset)
        self.category_mapping = {
            '1': 'Film & Animation',
            '2': 'Autos & Vehicles', 
            '10': 'Music',
            '15': 'Pets & Animals',
            '17': 'Sports',
            '19': 'Travel & Events',
            '20': 'Gaming',
            '22': 'People & Blogs',
            '23': 'Comedy',
            '24': 'Entertainment',
            '25': 'News & Politics',
            '26': 'Howto & Style',
            '27': 'Education',
            '28': 'Science & Technology'
        }
    
    def process(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        if 'category_id' in df.columns:
            df['category_name'] = df['category_id'].map(self.category_mapping)
            logger.info(f"Added category names to {table_name}")
        return df


# Example usage
if __name__ == "__main__":
    # Basic usage with default settings
    #loader = YouTubeDataLoader()
    #loader.load_to_database()
    
    # Advanced usage with custom preprocessors
    custom_preprocessors = [
        BasicCleanupPreprocessor(),
        ViewCountFilterPreprocessor(min_view_count=5000),
        #CategoryMappingPreprocessor()
    ]
    
    import json
    with open("db_config.json") as f:
        custom_db_config = json.load(f)

    advanced_loader = YouTubeDataLoader(
        db_config=custom_db_config,
        preprocessors=custom_preprocessors
    )
    
    # Load specific files
    advanced_loader.load_to_database()
    
    # Or load only video details
    #advanced_loader.load_to_database(file_type='video_details')