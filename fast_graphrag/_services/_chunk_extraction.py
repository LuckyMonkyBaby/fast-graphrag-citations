import re
from dataclasses import dataclass, field
from typing import Iterable, List, Set, Tuple

import xxhash

from fast_graphrag._types import (
    TChunk, 
    TDocument, 
    THash,
    TCitation
)
from fast_graphrag._utils import TOKEN_TO_CHAR_RATIO
from ._base import BaseChunkingService

# Regex pattern for detecting sentence boundaries
SENTENCE_PATTERN = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\。|\？|\！|\．)(\s+|$)'

DEFAULT_SEPARATORS = [
    # Paragraph and page separators
    "\n\n\n",
    "\n\n",
    "\r\n\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
]

def find_sentence_boundaries(text: str) -> List[Tuple[int, int]]:
    """
    Find the start and end offsets of all sentences in the text using regex.
    
    Args:
        text: The text to analyze
        
    Returns:
        A list of (start, end) tuples for each sentence
    """
    sentence_offsets: List[Tuple[int, int]] = []
    current_pos = 0
    
    # Split the text into sentences using regex
    sentence_boundaries = list(re.finditer(SENTENCE_PATTERN, text))
    
    if not sentence_boundaries:
        # If no sentence boundaries found, treat the whole text as one sentence
        if text.strip():
            return [(0, len(text))]
        return []
    
    # Process each sentence
    for i in range(len(sentence_boundaries)):
        # Get the sentence text including the ending punctuation
        if i < len(sentence_boundaries) - 1:
            next_start = sentence_boundaries[i+1].start()
            sentence_end = next_start
        else:
            sentence_end = len(text)
        
        # If there's content between current_pos and the sentence end
        if current_pos < sentence_end and text[current_pos:sentence_end].strip():
            sentence_offsets.append((current_pos, sentence_end))
        
        # Move to the position after this sentence boundary
        current_pos = sentence_boundaries[i].end()
    
    # Handle any remaining text after the last boundary
    if current_pos < len(text) and text[current_pos:].strip():
        sentence_offsets.append((current_pos, len(text)))
    
    return sentence_offsets

@dataclass
class DefaultChunkingServiceConfig:
    separators: List[str] = field(default_factory=lambda: DEFAULT_SEPARATORS)
    chunk_token_size: int = field(default=800)
    chunk_token_overlap: int = field(default=100)

@dataclass
class DefaultChunkingService(BaseChunkingService[TChunk]):
    """Default class for chunk extractor with citation tracking."""

    config: DefaultChunkingServiceConfig = field(default_factory=DefaultChunkingServiceConfig)

    def __post_init__(self):
        self._split_re = re.compile(f"({'|'.join(re.escape(s) for s in self.config.separators or [])})")
        self._chunk_size = self.config.chunk_token_size * TOKEN_TO_CHAR_RATIO
        self._chunk_overlap = self.config.chunk_token_overlap * TOKEN_TO_CHAR_RATIO

    async def extract(self, data: Iterable[TDocument]) -> Iterable[Iterable[TChunk]]:
        """Extract unique chunks from the given data."""
        chunks_per_data: List[List[TChunk]] = []

        for d in data:
            unique_chunk_ids: Set[THash] = set()
            extracted_chunks = await self._extract_chunks(d)
            chunks: List[TChunk] = []
            for chunk in extracted_chunks:
                if chunk.id not in unique_chunk_ids:
                    unique_chunk_ids.add(chunk.id)
                    chunks.append(chunk)
            chunks_per_data.append(chunks)

        return chunks_per_data

    async def _extract_chunks(self, data: TDocument) -> List[TChunk]:
        """Extract chunks from a document while tracking offsets."""
        # Sanitise input data:
        try:
            data.data = data.data.encode(errors="replace").decode()
        except UnicodeDecodeError:
            # Default to replacing all unrecognised characters with a space
            data.data = re.sub(r"[\x00-\x09\x11-\x12\x14-\x1f]", " ", data.data)

        if len(data.data) <= self._chunk_size:
            # Find sentence boundaries in the whole document
            sentence_offsets = find_sentence_boundaries(data.data)
            
            # For single chunk, use entire document range
            return [
                TChunk(
                    id=THash(xxhash.xxh3_64_intdigest(data.data)),
                    content=data.data,
                    metadata=data.metadata,
                    citation=TCitation(
                        start_offset=0,
                        end_offset=len(data.data),
                        sentence_offsets=sentence_offsets,
                    )
                )
            ]
        else:
            # Split text and track offsets
            chunks_with_offsets = self._split_text_with_offsets(data.data)
            return [
                TChunk(
                    id=THash(xxhash.xxh3_64_intdigest(chunk)),
                    content=chunk,
                    metadata=data.metadata,
                    citation=TCitation(
                        start_offset=start_offset,
                        end_offset=end_offset,
                        sentence_offsets=self._get_sentence_offsets_with_base(chunk, start_offset),
                    )
                )
                for chunk, start_offset, end_offset in chunks_with_offsets
            ]

    def _get_sentence_offsets_with_base(self, text: str, base_offset: int) -> List[Tuple[int, int]]:
        """
        Get sentence offsets within a chunk, adjusted for the chunk's base offset.
        
        Args:
            text: The chunk text
            base_offset: The offset of the chunk in the original document
            
        Returns:
            A list of sentence (start, end) offsets adjusted to the original document
        """
        # Find sentence boundaries within this chunk
        sentence_offsets: List[Tuple[int, int]] = find_sentence_boundaries(text)
        
        # Adjust offsets relative to the original document
        result: List[Tuple[int, int]] = [(start + base_offset, end + base_offset) for start, end in sentence_offsets]
        return result

    def _split_text_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text and track original character offsets."""
        splits_with_offsets: List[Tuple[str, int, int]] = []
        current_offset = 0
        
        # Split text and keep track of separator positions
        parts = self._split_re.split(text)
        
        for i, part in enumerate(parts):
            if not part:  # Skip empty parts
                continue
                
            # If it's a separator (odd indices), just update offset
            if i % 2 == 1:
                current_offset += len(part)
                continue
                
            # For content parts, track the offsets
            splits_with_offsets.append((part, current_offset, current_offset + len(part)))
            current_offset += len(part)
            
        return self._merge_splits_with_offsets(splits_with_offsets)

    def _merge_splits_with_offsets(
        self, 
        splits: List[Tuple[str, int, int]]
    ) -> List[Tuple[str, int, int]]:
        """Merge splits while preserving character offsets."""
        if not splits:
            return []

        merged_splits: List[Tuple[str, int, int]] = []
        current_chunk: List[Tuple[str, int, int]] = []
        current_length = 0

        for split, start_offset, end_offset in splits:
            split_length = len(split)
            
            if current_length + split_length <= self._chunk_size - (
                self._chunk_overlap if current_chunk else 0
            ):
                current_chunk.append((split, start_offset, end_offset))
                current_length += split_length
            else:
                if current_chunk:
                    chunk_text = "".join(part[0] for part in current_chunk)
                    chunk_start = current_chunk[0][1]
                    chunk_end = current_chunk[-1][2]
                    merged_splits.append((chunk_text, chunk_start, chunk_end))
                
                current_chunk = [(split, start_offset, end_offset)]
                current_length = split_length

        # Add the last chunk
        if current_chunk:
            chunk_text = "".join(part[0] for part in current_chunk)
            chunk_start = current_chunk[0][1]
            chunk_end = current_chunk[-1][2]
            merged_splits.append((chunk_text, chunk_start, chunk_end))

        if self._chunk_overlap > 0:
            return self._enforce_overlap_with_offsets(merged_splits)
        
        return merged_splits

    def _enforce_overlap_with_offsets(
        self, 
        chunks: List[Tuple[str, int, int]]
    ) -> List[Tuple[str, int, int]]:
        """Enforce overlap while preserving offset information."""
        result: List[Tuple[str, int, int]] = []
        
        for i, (chunk_text, start_offset, end_offset) in enumerate(chunks):
            if i == 0:
                result.append((chunk_text, start_offset, end_offset))
            else:
                # Calculate overlap with previous chunk
                prev_chunk = chunks[i-1][0]
                overlap_size = min(self._chunk_overlap, len(prev_chunk))
                overlap_text = prev_chunk[-overlap_size:]
                
                # Create new chunk with overlap
                new_chunk = overlap_text + chunk_text
                # Adjust start offset to account for overlap
                new_start = end_offset - len(new_chunk)
                
                result.append((new_chunk, new_start, end_offset))
        
        return result