import logging
import json
import os
from src.extract import Extract
from src.transform import Transform
from src.load import YouTubeDataLoader

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('main.log')
    ]
)

logger = logging.getLogger(__name__)


class YoutubeETLPipeline:
    from typing import Dict
    def __init__(self, extraction_config: Dict[str, str], 
                 db_config: Dict[str, str], 
                 max_pages: int,
                 GOOGLE_API_KEY: str
                 ) -> None:
        """
        Args:
            extraction_config (Dict[str, str]): Configuration for data extraction.
            db_config (Dict[str, str]): Configuration for the database connection.
            max_pages (int): Maximum number of pages to extract (Each page contains 5 videos)
            GOOGLE_API_KEY (str): Google API key for authentication.
        """


        self.extraction_config = extraction_config
        self.db_config = db_config
        self.max_pages = max_pages
        self.GOOGLE_API_KEY = GOOGLE_API_KEY

    def run(self) -> None:
        # Extract
        try:
            logger.info("Starting data extraction...")
            extractor = Extract(self.extraction_config, max_pages=self.max_pages, GOOGLE_API_KEY=self.GOOGLE_API_KEY)
            raw_data = extractor.extract_all()
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            return

        # Transform
        try:
            transformer = Transform("data/raw/extract/")
            trends, videos = transformer.process_all()
            transformer.save_processed_data("data/processed/", format="parquet")
        except Exception as e:
            logger.error(f"Error during transformation: {e}")
            return
        # Load
        try:
            logger.info("Starting data loading...")
            loader = YouTubeDataLoader(processed_data_path="data/processed/", db_config=self.db_config)
            loader.load_to_database()
            logger.info("Data loading completed successfully.")
        except ConnectionRefusedError:
            logger.error("Connection to PostgreSQL server refused. Please ensure the server is running and the credentials are correct.")
        except FileNotFoundError:
            logger.error(f"Processed data directory data/processed/ not found.")
        except Exception as e:
            logger.error(f"Error during loading: {e}")


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv
    load_dotenv()


    extraction_config = None
    try:
        with open("src/extraction_config.json") as f:
            extraction_config = json.load(f)
    except FileNotFoundError:
        print("Extraction configuration file not found.")
        extraction_config = json.loads(os.getenv("EXTRACTION_CONFIG", "{}"))

    db_config = None
    try:
        with open("db_config.json") as f:
            db_config = json.load(f)
    except FileNotFoundError:
        print("Database configuration file not found.")
        db_config = json.loads(os.getenv("DB_CONFIG", "{}"))


    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY environment variable not set.")
        exit()

    parser = argparse.ArgumentParser(description="YouTube ETL Pipeline")
    parser.add_argument("--max_pages", type=int, default=15, help="Maximum number of pages to extract")
    args = parser.parse_args()

    pipeline = YoutubeETLPipeline(
        extraction_config=extraction_config,
        db_config=db_config,
        max_pages=args.max_pages,
        GOOGLE_API_KEY=GOOGLE_API_KEY
    )
    pipeline.run()
