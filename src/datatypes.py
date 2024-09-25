from collections import namedtuple

LangIdx = namedtuple('LangIdx', ['src_idx', 'dest_idx', 'latent_idx'],
                     defaults=[None, None, None])

TokenizedSuffixesResult = namedtuple('TokenizedSuffixesResult', 
                                     ['input_ids', 'attention_mask', 'indices'], 
                                     defaults=[None, None, None])