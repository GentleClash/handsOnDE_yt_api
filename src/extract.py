# Extract

import logging
import requests
import os
from typing import Optional, Dict, Any, List
import json
from datetime import datetime
from urllib import parse
import traceback

logger = logging.getLogger(__name__)
class Extract:
    """
    This class is responsible for extracting raw data from the YouTube API.
    """

    def __init__(self, config_file: str | Dict , 
                 GOOGLE_API_KEY: Optional[str] = None, 
                 BASE_URL: Optional[str] = None, 
                 max_pages: Optional[int] = None):
        """
        Initializes the Extract class with the given configuration.

        Args:
            config_file (str | Dict): The path to the configuration file or a dictionary containing configuration.
            GOOGLE_API_KEY (Optional[str]): The Google API key to use for authentication.
            BASE_URL (Optional[str]): The base URL for the YouTube API.
            max_pages (Optional[int]): The maximum number of pages to fetch for certain API calls.
        """


        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            logger.warning("dotenv module not found, skipping environment variable loading")

        if not GOOGLE_API_KEY:
            self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
            if not self.GOOGLE_API_KEY:
                logger.error("Missing GOOGLE_API_KEY environment variable")
                raise ValueError("Missing GOOGLE_API_KEY environment variable")
        else:
            self.GOOGLE_API_KEY = GOOGLE_API_KEY

        if isinstance(config_file, str):
            with open(config_file, "r") as f:
                config = json.load(f)
        else:
            config = config_file

        if not BASE_URL:
            self.BASE_URL = config.get("BASE_URL", "https://www.googleapis.com/youtube/v3")
        else:
            self.BASE_URL = BASE_URL

        self.endpoint = config.get("endpoint", "/videos")
        self.DIR = config.get("DIR", "data/raw/extract/")
        self.trend_list_parameters = config.get("TREND_LIST_PARAMETERS", {
                                                                    "part": "snippet",
                                                                    "regionCode": "IN",
                                                                    "chart": "mostPopular"
                                                                })
        self.video_details_parameter = config.get("VIDEO_DETAILS_PARAMETER", {
                                                            "part": "snippet,contentDetails,statistics",
                                                            "id": ""
                                                        })
        logger.info("Initialized Extract class with config: %s", config)

        # Url for trend list
        self._trend_list = None
        self._video_ids : List[str] = []
        self.max_pages = max_pages if max_pages is not None else 1
        self.complete_file_name = None

    def _get_url(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Returns the full URL for a given API endpoint and query parameters.

        Args:
            endpoint (str): The API endpoint to call.
            params (Optional[Dict[str, Any]]): Query parameters for the API call.

        Returns:
            str: The full URL for the API call.
        """
        if params is None:
            params = {}
        params['key'] = self.GOOGLE_API_KEY
        url = f"{self.BASE_URL}{endpoint}?{parse.urlencode(params)}"
        return url

    def get_trend_list(self) -> List[Dict[str, Any]]:
        """
        Fetches the trend list from the API if it hasn't been fetched already.

        Returns:
            List[Dict[str, Any]]: The trend list data.
        """
        if self._trend_list is None:
            logger.info("Fetching trend list from API")
            self._trend_list = self.get_response(self.endpoint, self.trend_list_parameters, dump=True, max_pages=self.max_pages, base_file_name="trend_list")
        return self._trend_list

    def _extract_video_ids_from_trend_list(self, trend_list: List[Dict[str, Any]]) -> List[str]:
        """
        Extract video IDs from the trend list response.

        Args:
            trend_list (List[Dict[str, Any]]): The trend list data from the API response.

        Returns:
            List[str]: A list of extracted video IDs.
        """
        video_ids = []
        for page_data in trend_list:
            if 'items' in page_data:
                for item in page_data['items']:
                    if 'id' in item:
                        video_ids.append(item['id'])
        logger.info(f"Extracted {len(video_ids)} video IDs from trend list")
        return video_ids

    def get_video_ids(self) -> List[str]:
        """
        Get video IDs from the trend list, fetching if necessary.

        Returns:
            List[str]: A list of video IDs.
        """
        if not self._video_ids:
            trend_list = self.get_trend_list()
            self._video_ids = self._extract_video_ids_from_trend_list(trend_list)
        return self._video_ids

    def _batch_video_ids_params(self, base_params: Dict[str, Any], video_ids: List[str]) -> Dict[str, Any]:
        """
        Create parameters for batched video details request.
        Youtube accepts batch video IDs separated by commas.

        Args:
            base_params (Dict[str, Any]): The base parameters for the API request.
            video_ids (List[str]): The list of video IDs to include in the request.

        Returns:
            Dict[str, Any]: The updated parameters for the API request.
        """
        params = base_params.copy()
        params["id"] = ",".join(video_ids)
        return params

    def get_video_details(self, batch_size: int = 50) -> List[Dict[str, Any]]:
        """
        Get detailed information for all trending videos.

        Args:
            batch_size (int, optional): The number of videos to fetch in each batch. Defaults to 50.

        Returns:
            List[Dict[str, Any]]: A list of video details.
        """
        video_ids = self.get_video_ids()
        if not video_ids:
            logger.warning("No video IDs found to fetch details for")
            return []

        logger.info(f"Fetching details for {len(video_ids)} videos in batches of {batch_size}")

        all_video_details = []
        
        # Process video IDs in batches
        for i in range(0, len(video_ids), batch_size):
            batch_ids = video_ids[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            logger.info(f"Fetching batch {batch_num} with {len(batch_ids)} videos")
            
            # Use get_response for each batch
            params = self._batch_video_ids_params(self.video_details_parameter.copy(), batch_ids)

            batch_response = self.get_response(
                endpoint="/videos",
                params=params,
                dump=True,
                max_pages=1,
                base_file_name=f"video_details_batch_{batch_num}"
            )
            
            all_video_details.extend(batch_response)
        
        return all_video_details

    def extract_all(self, save_dir : Optional[str] = None) -> Dict[str, Any]:
        """
        Complete extraction process: get trending videos and their detailed information.

        Args:
            save_dir (Optional[str]): Directory to save the extraction results. <mark style="background-color: #FF0000;">CAUTION: This overrides the self.DIR</mark>

        Returns:
            Dict[str, Any]: The complete extraction result containing {‘trend_list’, ‘video_details’, ‘extraction_timestamp’, ‘total_videos’}
        """
        logger.info("Starting complete extraction process")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.DIR = save_dir
            logger.warning(f"Overriding self.DIR with: {self.DIR}")

        # Get trending list
        trend_list = self.get_trend_list()
        logger.info(f"Fetched {len(trend_list)} pages of trending data")
        
        # Get video details
        video_details = self.get_video_details()
        logger.info(f"Fetched {len(video_details)} batches of video details")
        
        # Combine and return results
        result = {
            'trend_list': trend_list,
            'video_details': video_details,
            'extraction_timestamp': datetime.now().isoformat(),
            'total_videos': len(self.get_video_ids())
        }
        
        # Save complete result
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        complete_filename = f"{self.DIR}complete_extraction_{timestamp}.json"
        
        with open(complete_filename, 'w') as f:
            json.dump(result, f, indent=4)
        logger.info(f"Saved complete extraction to: {complete_filename}")
        
        return result
    

    def get_response(self, endpoint: str, params: Optional[Dict[str, str]] = None, dump: bool = True, max_pages: int = 1, base_file_name: Optional[str] = None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        page_count = 0
        next_page_token = None

        if base_file_name:
            base_name_for_dump: str = self.DIR + base_file_name.replace('/', '_')
            path = os.path.dirname(base_name_for_dump)
            os.makedirs(path, exist_ok=True)
            base_name_for_dump = f"{base_name_for_dump}_response"

        else:
            base_name_for_dump: str = f"{self.DIR}{endpoint.replace('/', '_')}_response"
            os.makedirs(os.path.dirname(base_name_for_dump), exist_ok=True)

        while page_count < max_pages:
            if next_page_token:
                params = params.copy() if params else {}
                params['pageToken'] = next_page_token
            url = self._get_url(endpoint, params)

            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code} - {response.text}")
            data = response.json()
            results.append(data)

            if dump:
                print(f"Dumping page {page_count + 1}...")
                page_suffix: str = f"_page_{page_count+1}"
                timestamp: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                current_file_name: str = f"{base_name_for_dump}{page_suffix}_{timestamp}.json"

                with open(current_file_name, 'w') as f:
                    json.dump(data, f, indent=4)
                print(f"Dumped to: {current_file_name}")

            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                break
            page_count += 1

        return results


# Example usage
if __name__ == "__main__":
    try:
        config_file_path = "extraction_config.json"
        extractor = Extract(config_file=config_file_path, max_pages=15)

        # Option 1: Run complete extraction
        results = extractor.extract_all(save_dir="../data/raw/extract/")
        print(f"Extraction complete! Found {results['total_videos']} videos")
        
        # Option 2: Step by step
        # trend_list = extractor.get_trend_list()
        # video_ids = extractor.get_video_ids()
        # video_details = extractor.get_video_details()
        
    except Exception as e:
        traceback.print_exc()
        raise
