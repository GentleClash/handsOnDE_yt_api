import logging
import json
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

# Extract
extraction_config = None
with open("src/extraction_config.json") as f:
    extraction_config = json.load(f)

extractor = Extract(extraction_config, max_pages=15)
raw_data = extractor.extract_all()

# Transform
transformer = Transform( "data/raw/extract/")
trend_ids, videos = transformer.process_all()

transformer.save_processed_data("data/processed/", format="parquet")
print(f"Processed {len(videos)} videos with {len(trend_ids)} unique trend IDs.")


# Load

with open("db_config.json") as f:
    custom_db_config = json.load(f)

try:
    loader = YouTubeDataLoader(processed_data_path="data/processed/", db_config=custom_db_config)
    loader.load_to_database()

# pg server not running
except ConnectionRefusedError:
    logger.error("Connection to PostgreSQL server refused. Please ensure the server is running and the credentials are correct.")

# directory not found
except FileNotFoundError:
    logger.error(f"Processed data directory data/processed/ not found.")
# other errors
except Exception as e:
    logger.error(f"Error loading data: {e}")


