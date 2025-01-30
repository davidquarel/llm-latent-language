from typing import Optional, Union
from torch import Tensor
from dataclasses import dataclass


@dataclass
class TokenizedSuffixesResult:
    input_ids: Optional[Tensor] = None
    attention_mask: Optional[Tensor] = None
    indices: Optional[Tensor] = None

    def __len__(self):
        return len(self.input_ids)
    
    def __post_init__(self):
        assert self.input_ids is None or self.attention_mask is None or self.input_ids.shape[0] == self.attention_mask.shape[0], "input_ids and attention_mask must have the same batch size"

    def __getitem__(self, idx: Union[int, slice]) -> 'TokenizedSuffixesResult':
        """Support integer and slice-based indexing."""
        return TokenizedSuffixesResult(
            input_ids=self.input_ids[idx] if self.input_ids is not None else None,
            attention_mask=self.attention_mask[idx] if self.attention_mask is not None else None,
            indices=self.indices[idx] if self.indices is not None else None
        )

    def to(self, *args, **kwargs):
        """Move all tensor attributes to the specified device/dtype."""
        return TokenizedSuffixesResult(
            input_ids=self.input_ids.to(*args, **kwargs) if self.input_ids is not None else None,
            attention_mask=self.attention_mask.to(*args, **kwargs) if self.attention_mask is not None else None,
            indices=self.indices.to(*args, **kwargs) if self.indices is not None else None
        )
        
    def to_tuple(self):
        return tuple([t for t in [self.input_ids, self.attention_mask, self.indices] if t is not None])