from typing import List

from collections import OrderedDict
from gensim.parsing.preprocessing import *

from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest
from src.selector.selector import Selector


FILTERS = [
    stem_text,  # lower cases and stems
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    remove_stopwords,
]


class DedupingSelector(Selector):
    def select(
        self,
        request: SimilarTagRequest,
        candidates: List[SimilarTagCandidate]
    ) -> List[SimilarTagCandidate]:
        canonical = OrderedDict()
        for candidate in candidates:
            p_tag = ''.join(preprocess_string(candidate.tag, FILTERS))
            if p_tag not in canonical:
                canonical[p_tag] = candidate
            else:
                for k, v in candidate.features.items():
                    if k not in canonical[p_tag].features:
                        canonical[p_tag].features[k] = v

        return list(canonical.values())[:request.limit]
