from pathway import UDF
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, TypedDict

import numpy as np
import spacy
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
from llama_index.core.schema import TextNode, BaseNode

@dataclass
class SentenceCombination(TypedDict):
    sentence: str
    index: int
    combined_sentence: str
    combined_sentence_embedding: list[float]


BreakpointThresholdType = Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
]


BREAKPOINT_DEFAULTS: dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
    "gradient": 95,
}


def split_by_spacy(text: str) -> List[str]:
    """
    Split text into sentences using spaCy.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


class SemanticSplitter(SemanticSplitterNodeParser, UDF):

    breakpoint_threshold_type: BreakpointThresholdType
    breakpoint_threshold_amount: Optional[float]
    sentence_splitter: Callable[[str], List[str]] = split_by_spacy
    minimum_chunk_size: int = 60

    def __init__(
        self,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        sentence_splitter: Callable[[str], List[str]] = split_by_spacy,
        minimum_chunk_size: int = 60,
    ):
        super().__init__()
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.sentence_splitter = sentence_splitter
        self.minimum_chunk_size = minimum_chunk_size
        
    def __wrapped__(self, txt: str, **kwargs) -> list[tuple[str, dict]]:
        documet = Document(text=txt)
        chunks = self.get_nodes_from_documents([documet])
        # make shitty chunk dict tuple next to match pathway
        splits = []
        for chunk in chunks:
            splits.append((chunk.get_text(), {}))
        
        return splits

    def _get_threshold(self, distances: list[float]) -> float:
        if self.breakpoint_threshold_amount is None:
            self.breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[
                self.breakpoint_threshold_type
            ]

        if self.breakpoint_threshold_type == "percentile":
            breakpoint_distance_threshold = np.percentile(
                distances, self.breakpoint_threshold_amount
            )
        elif self.breakpoint_threshold_type == "standard_deviation":
            breakpoint_distance_threshold = np.mean(distances) + (
                self.breakpoint_threshold_amount * np.std(distances)
            )
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1

            breakpoint_distance_threshold = (
                np.mean(distances) + self.breakpoint_threshold_amount * iqr
            )
        elif self.breakpoint_threshold_type == "gradient":
            distance_gradient = np.gradient(distances, np.arange(len(distances)))
            breakpoint_distance_threshold = np.percentile(
                distance_gradient, self.breakpoint_threshold_amount
            )

        return float(breakpoint_distance_threshold)

    def _build_node_chunks(
        self, sentences: list[SentenceCombination], distances: list[float]
    ) -> list[str]:
        chunks = []
        if len(distances) > 0:
            breakpoint_distance_threshold = self._get_threshold(distances)

            indices_above_threshold = [
                i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
            ]
            start_index = 0

            for index in indices_above_threshold:
                group = sentences[start_index : index + 1]
                combined_text = "".join([d["sentence"] for d in group])
                chunks.append(combined_text)

                start_index = index + 1

            if start_index < len(sentences):
                combined_text = "".join(
                    [d["sentence"] for d in sentences[start_index:]]
                )
                chunks.append(combined_text)

            # Check token count and merge with the previous chunk if necessary
            i = 1
            nlp = spacy.load("en_core_web_sm")
            while i < len(chunks):
                doc = nlp(chunks[i])
                if len(doc) < self.minimum_chunk_size:
                    # Merge with the previous chunk
                    chunks[i - 1] = chunks[i - 1] + chunks[i]
                    del chunks[i]
                else:
                    i += 1
        else:
            chunks = [" ".join([s["sentence"] for s in sentences])]

        return chunks
