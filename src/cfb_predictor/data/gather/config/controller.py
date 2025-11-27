from typing import List, Optional, Any, Dict
from collections import UserList
from pathlib import Path
import requests
from urllib.parse import urljoin
import yaml
import json
from tqdm import tqdm
import logging

from cfb_predictor.data.gather.config.model import RequestConfig, RequestConfigList
from cfb_predictor.core.logging import tqdm_logging_redirect

current_dir = Path(__file__).parent
logger = logging.getLogger(__name__)

class RequestController:
    def __init__(
        self,
        api_key: Optional[str] = None,
        config_folder : Optional[Path] = None,
        output_folder: Optional[Path] = None
    ):
        self.api_key = RequestController._resolve_api_key(api_key)
        self.config_folder = config_folder or current_dir / "configs"
        self.output_folder = output_folder or current_dir.parent / "output"

    @property
    def _base_url(self) -> str:
        return "https://api.collegefootballdata.com/"
    
    @property
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _load_config_set(self, config_name: str) -> List[RequestConfig]:
        config_path = self.config_folder / f"{config_name}.yml"
        with open(config_path, "r") as file:
            raw_configs = yaml.safe_load(file)
        
        request_config_list = RequestConfigList(**raw_configs)
        return RequestList(request_config_list.all_requests)

    def _load_all_configs(self) -> List[RequestConfig]:
        all_configs = []
        for config_file in self.config_folder.glob("*.yml"):
            config_name = config_file.stem
            configs = self._load_config_set(config_name)
            all_configs.extend(configs)
        return RequestList(all_configs)
    
    @staticmethod
    def _resolve_api_key(api_key: Optional[str] = None) -> str:
        if api_key is not None:
            return api_key
        
        from dotenv import load_dotenv
        import os
        load_dotenv()
        env_api_key = os.getenv("API_KEY")
        if env_api_key is None:
            raise ValueError("API key must be provided either as an argument or in the environment variable")
        return env_api_key
        
    
    def retrieve_data(self, config_name: str = "all", overwrite: bool = False) -> None:
        if config_name == "all":
            request_configs = self._load_all_configs()
        else:
            request_configs = self._load_config_set(config_name)

        self.output_folder.mkdir(parents=True, exist_ok=True)

        with tqdm_logging_redirect():
            pbar = tqdm(total=len(request_configs), desc="Processing requests", position=0)
            for request_config in request_configs:
                pbar.set_description(f"Processing {request_config.path}")
                output = self._process_single_request(request_config, overwrite)
                pbar.update(1)
                
                if output == {}: continue

                for advanced_stats in request_config.advanced_stats:
                    for advanced_stat_request in RequestConfig.from_advanced_stats(advanced_stats, output):
                        pbar.set_description(f"Processing {advanced_stat_request.path}")
                        _ = self._process_single_request(advanced_stat_request, overwrite, included_data=advanced_stat_request.params)
                    pbar.update(1)
        
        self.report_info()
        
    def report_info(self):
        url = urljoin(self._base_url, "info/")
        response = requests.get(url, headers=self._headers)
        logger.info(f"Info - {response.text}") 

    def _process_single_request(
            self, 
            request_config: RequestConfig, 
            overwrite: bool, 
            included_data: Dict = {}
        ) -> Any | None:
        output_path = self.output_folder / request_config.file_name
        if output_path.exists() and not overwrite:
            logger.info(f"File {output_path} already exists. Skipping download.")
            with open(output_path, "r") as file:
                output = json.load(file)
            return output # In case we need this for advanced stats that haven't already been retrieved

        url = urljoin(self._base_url, request_config.path)
        try:
            response = requests.get(url, params=request_config.params, headers=self._headers)
            response.raise_for_status()
            output = response.json()
        except Exception as e:
            logger.warning(f"Failed to get response for request {request_config}: {e}")
            return {}
 

        if isinstance(output, dict):
            output.update(included_data)
        elif included_data != {}:
            logger.warning(f"Couldn't include additional data {included_data} for request {request_config} "
                           f"since the response is type {type(response)}.")
            
        with open(output_path, "w") as file:
            json.dump(output, file, indent=4)
        logger.debug(f"Data saved to {output_path}")

        return output
    
class RequestList(UserList):
    def __init__(self, requests_list: List[RequestConfig]):
        super().__init__(requests_list)
    
    def __len__(self) -> int:
        total_len = len(self.data)
        for request in self.data:
            total_len += len(request.advanced_stats)
        return total_len
    