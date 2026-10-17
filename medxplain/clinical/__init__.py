"""Clinical-workspace analysis of a single discharge note.

* :mod:`~medxplain.clinical.text` - offset-preserving section/sentence parsing
* :mod:`~medxplain.clinical.concepts` - transparent keyword concept detection
* :mod:`~medxplain.clinical.extraction` - case summary, documentation review, follow-up
* :mod:`~medxplain.clinical.similar_cases` - retrieval of comparable historical cases
"""

from medxplain.clinical.extraction import analyse_note
from medxplain.clinical.similar_cases import CaseLibrary

__all__ = ["analyse_note", "CaseLibrary"]
