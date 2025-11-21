from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from urllib.parse import urljoin
import yaml
import json
from tqdm import tqdm
import logging

current_dir = Path(__file__).parent
logger = logging.getLogger(__name__)

@dataclass
class RequestConfig:
    path: str
    params: Dict[str, Any]

    @property
    def file_name(self) -> str:
        param_str = "_".join(f"{key}-{value}" for key, value in self.params.items())
        return f"{self.path.replace('/', '_')}_{param_str}.json"

@dataclass
class RequestConfigList:
    path: str
    params: Dict[str, List[Any]]

    def parse_requests(self) -> List[RequestConfig]:
        keys = list(self.params.keys())
        values_list = list(zip(*self.params.values()))
        return [RequestConfig(path=self.path, params=dict(zip(keys, values))) for values in values_list]
    
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

    def _load_config_set(self, config_name: str) -> List[RequestConfig]:
        config_path = self.config_folder / f"{config_name}.yml"
        with open(config_path, "r") as file:
            raw_configs = yaml.safe_load(file)
        
        request_config_list = RequestConfigList(**raw_configs)
        return request_config_list.parse_requests()

    def _load_all_configs(self) -> List[RequestConfig]:
        all_configs = []
        for config_file in self.config_folder.glob("*.yml"):
            config_name = config_file.stem
            configs = self._load_config_set(config_name)
            all_configs.extend(configs)
        return all_configs
    
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
        
        base_url = "https://api.collegefootballdata.com/"

        self.output_folder.mkdir(parents=True, exist_ok=True)

        headers = {"Authorization": f"Bearer {self.api_key}"}

        for request_config in tqdm(request_configs, desc="Retrieving data", position=0):
            output_path = self.output_folder / request_config.file_name
            if output_path.exists() and not overwrite:
                logger.info(f"File {output_path} already exists. Skipping download.")
                continue

            url = urljoin(base_url, request_config.path)
            response = requests.get(url, params=request_config.params, headers=headers)
            response.raise_for_status()

            with open(output_path, "w") as file:
                json.dump(response.json(), file, indent=4)
            logger.debug(f"Data saved to {output_path}")


