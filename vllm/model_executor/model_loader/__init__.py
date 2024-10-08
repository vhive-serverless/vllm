from typing import Optional

from torch import nn

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig, LiquidConfig)
from vllm.model_executor.model_loader.loader import (BaseModelLoader,
                                                     get_model_loader,
                                                     ServerlessLLMLoader)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture)


def get_model(*, model_config: ModelConfig, load_config: LoadConfig,
              device_config: DeviceConfig, parallel_config: ParallelConfig,
              scheduler_config: SchedulerConfig,
              lora_config: Optional[LoRAConfig],
              vision_language_config: Optional[VisionLanguageConfig],
              cache_config: CacheConfig,
              liquid_config: Optional[LiquidConfig]) -> nn.Module:
    loader = get_model_loader(load_config)
    if isinstance(loader, ServerlessLLMLoader):
        return loader.load_model(model_config=model_config,
                                device_config=device_config,
                                lora_config=lora_config,
                                vision_language_config=vision_language_config,
                                parallel_config=parallel_config,
                                scheduler_config=scheduler_config,
                                cache_config=cache_config,
                                )
    else:
        return loader.load_model(model_config=model_config,
                                device_config=device_config,
                                lora_config=lora_config,
                                vision_language_config=vision_language_config,
                                parallel_config=parallel_config,
                                scheduler_config=scheduler_config,
                                cache_config=cache_config,
                                liquid_config = liquid_config,
                                )


__all__ = [
    "get_model", "get_model_loader", "BaseModelLoader",
    "get_architecture_class_name", "get_model_architecture"
]
