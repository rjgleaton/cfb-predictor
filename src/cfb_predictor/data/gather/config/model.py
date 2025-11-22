from dataclasses import dataclass
from pydantic import BaseModel, field_validator

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from itertools import product
from functools import cached_property
from collections import Counter

current_dir = Path(__file__).parent
logger = logging.getLogger(__name__)


#TODO Find a way to generically allow advanced stats to be collected like game/box/advanced
# Where the parameter is a game id from the previously collected data.
@dataclass
class KeyMapping:
    # Class that holds which key mappings from the output of one request to the input
    # of another request
    key_in: str # Param name to pass into advanced stats request
    key_out: str # Output key name from response

    def __eq__(self, other):
        if isinstance(other, KeyMapping):
            return (self.key_in == other.key_in) and (self.key_out == other.key_out)
        return False
    
    def __hash__(self):
        return hash((self.key_in, self.key_out))

@dataclass
class AdvancedStats:
    path: str
    key_mapping: List[KeyMapping]

    def __eq__(self, other):
        if isinstance(other, AdvancedStats):
            return (self.path == other.path) and (Counter(self.key_mapping) == Counter(other.key_mapping))
    
    def __hash__(self):
        return hash((self.path, *self.key_mapping))

class RequestConfig(BaseModel):
    path: str
    params: Dict[str, Any] = {}
    advanced_stats: List[AdvancedStats] = []

    @property
    def file_name(self) -> str:
        param_str = "_".join(f"{key}-{value}" for key, value in self.params.items())
        return f"{self.path.replace('/', '_')}_{param_str}.json"
    
    @staticmethod
    def from_advanced_stats(advanced_stats: AdvancedStats, response: Any) -> List["RequestConfig"]:
        request_configs = []

        def retrieve_params(response_dict: Dict):
            params = {}
            for key_mapping in advanced_stats.key_mapping:
                params[key_mapping.key_in] = response_dict[key_mapping.key_out]
            return params
        
        if isinstance(response, list):
            for response_dict in response:
                params = retrieve_params(response_dict)
                request_configs.append(RequestConfig(path=advanced_stats.path, params=params))
        else:
            params = retrieve_params(response)
            request_configs.append(RequestConfig(path=advanced_stats.path, params=params))

        return request_configs
    

class EndpointConfigSet(BaseModel):
    path: str
    params: Dict[str, List[Any]] = {}
    advanced_stats: List[AdvancedStats] = []

    def parse_requests(self) -> List[RequestConfig]:
        # Create a list of RequestConfigs for all parameter combinations
        param_keys = list(self.params.keys())
        param_values = [self.params[key] for key in param_keys]
        all_param_combinations = product(*param_values)
        return [
            RequestConfig(path=self.path, params=dict(zip(param_keys, values)), advanced_stats=self.advanced_stats) 
            for values in all_param_combinations
        ]


class RequestConfigList(BaseModel):
    configs: List[EndpointConfigSet]

    @cached_property
    def all_requests(self) -> List[RequestConfig]:
        all_requests = []
        for config_set in self.configs:
            all_requests.extend(config_set.parse_requests())
        return all_requests

    @field_validator("configs", mode="before")
    def validate_configs(cls, v):
        if isinstance(v, dict):
            v = [EndpointConfigSet(**item) for item in v['configs']]
        return v

        

