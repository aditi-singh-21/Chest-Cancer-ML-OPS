from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen = True)
class DataIngestionConfig:
    root_dir : Path
    source_url : str
    local_data_file : Path
    unzip_dir : Path
    
@dataclass(frozen = True)
class PrepareBaseModelConfig:
    root_dir : Path
    base_model_path : Path
    updated_base_model_path : Path
    image_size_params : list
    learning_rate_params : float
    include_top_params : bool
    weights_params : str
    classes_params : int
    