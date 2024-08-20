from typing import Dict, Tuple
from attr import dataclass
import torch
from torch import Size
import re

SPLIT = ';'

@dataclass
class MetaData:
    offset: int
    length: int
    shape: Size 

    @staticmethod
    def str2size(size_str:str) -> Size:
        # Use regular expression to find the content inside parentheses
        match = re.search(r'\[([0-9, ]+)\]', size_str)
        if match:
            # Extract the matched content and split by comma to get the dimensions
            size_str = match.group(1)
            size_list = [int(dim) for dim in size_str.split(',')]
            return Size(size_list)
        else:
            raise ValueError(f"No size found in the string: {size_str}")

    def __str__(self) -> str:
        return f"{self.offset}{SPLIT}{self.length}{SPLIT}{self.shape}"

    @classmethod
    def from_str(cls, meta_data_str:str):
        try:
            str_list = meta_data_str.split(SPLIT)
            assert len(str_list) == 3
            offset = int(str_list[0])
            length = int(str_list[1])
            shape = MetaData.str2size(str_list[2])
            return cls(offset, length, shape)
        except:
            raise ValueError(f"{meta_data_str} cannot convert to meta data!")

def main():
    tensor = torch.empty(10)
    meta_data = MetaData(0, 10, tensor.shape)
    meta_data_str = str(meta_data)
    print(meta_data_str)

    converted_meta_data = MetaData.from_str(meta_data_str)
    print(converted_meta_data)

if __name__ == "__main__":
    main()