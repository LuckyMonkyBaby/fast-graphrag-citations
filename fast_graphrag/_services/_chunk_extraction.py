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

# Existing default separators
DEFAULT_SEPARATORS: List[str] = [
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

# Regex pattern for extracting sentences
# This handles common sentence ending punctuation in English and some other languages
SENTENCE_PATTERN: str = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\。|\？|\！|\．)(\s+|$)'

def extract_sentences(text: str) -> List[str]:
    """Extracts sentences using regex instead of spaCy."""
    # Split by sentence boundary pattern
    raw_sentences: List[str] = re.split(SENTENCE_PATTERN, text)
    processed_sentences: List[str] = []
    
    i: int = 0
    while i < len(raw_sentences):
        current_sentence: str = raw_sentences[i]
        current_stripped: str = current_sentence.strip()
        
        if current_stripped:  # If the sentence has content
            # If there's a corresponding whitespace, include it
            if i + 1 < len(raw_sentences):
                next_sentence: str = raw_sentences[i + 1]
                next_stripped: str = next_sentence.strip()
                
                if not next_stripped:
                    processed_sentences.append(current_sentence + next_sentence)
                    i += 2
                    continue
            
            processed_sentences.append(current_sentence)
        
        i += 1
    
    # Final filtering for any empty sentences
    result: List[str] = []
    for sentence in processed_sentences:
        stripped_sentence: str = sentence.strip()
        if stripped_sentence:
            result.append(stripped_sentence)
    
    return result

@dataclass
class DefaultChunkingServiceConfig:
    separators: List[str] = field(default_factory=lambda: DEFAULT_SEPARATORS)
    chunk_token_size: int = field(default=800)
    chunk_token_overlap: int = field(default=100)

@dataclass
class DefaultChunkingService(BaseChunkingService[TChunk]):
    """Default class for chunk extractor with citation tracking."""

    config: DefaultChunkingServiceConfig = field(default_factory=DefaultChunkingServiceConfig)

    def __post_init__(self) -> None:
        self._split_re = re.compile(f"({'|'.join(re.escape(s) for s in self.config.separators or [])})")
        self._chunk_size: int = self.config.chunk_token_size * TOKEN_TO_CHAR_RATIO
        self._chunk_overlap: int = self.config.chunk_token_overlap * TOKEN_TO_CHAR_RATIO

    async def extract(self, data: Iterable[TDocument]) -> Iterable[Iterable[TChunk]]:
        """Extract unique chunks from the given data."""
        chunks_per_data: List[List[TChunk]] = []

        for d in data:
            unique_chunk_ids: Set[THash] = set()
            extracted_chunks: List[TChunk] = await self._extract_chunks(d)
            chunks: List[TChunk] = []
            
            for chunk in extracted_chunks:
                if chunk.id not in unique_chunk_ids:
                    unique_chunk_ids.add(chunk.id)
                    chunks.append(chunk)
            
            chunks_per_data.append(chunks)

        return chunks_per_data

    async def _extract_chunks(self, data: TDocument) -> List[TChunk]:
        """Extract chunks from a document with proper citation tracking."""
        # Sanitise input data:
        cleaned_text: str = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", data.data)
        data.data = cleaned_text

        if len(data.data) <= self._chunk_size:
            # Extract sentence offsets for the document
            sentence_offsets: List[Tuple[int, int]] = self._extract_sentence_offsets(data.data)
            
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
            chunks_with_offsets: List[Tuple[str, int, int]] = self._split_text_with_offsets(data.data)
            result: List[TChunk] = []
            
            for chunk, start_offset, end_offset in chunks_with_offsets:
                sentence_offsets_with_base: List[Tuple[int, int]] = self._extract_sentence_offsets_with_base(
                    chunk, start_offset
                )
                
                chunk_obj: TChunk = TChunk(
                    id=THash(xxhash.xxh3_64_intdigest(chunk)),
                    content=chunk,
                    metadata=data.metadata,
                    citation=TCitation(
                        start_offset=start_offset,
                        end_offset=end_offset,
                        sentence_offsets=sentence_offsets_with_base,
                    )
                )
                result.append(chunk_obj)
                
            return result

    def _extract_sentence_offsets(self, text: str) -> List[Tuple[int, int]]:
        """Extract sentence offsets using regex search."""
        offsets: List[Tuple[int, int]] = []
        current_pos: int = 0
        
        sentences: List[str] = extract_sentences(text)  # Get sentences using regex
        
        for sentence in sentences:
            stripped_sentence: str = sentence.strip()
            if not stripped_sentence:
                continue
                
            # Escape special regex characters in the sentence
            escaped_sentence: str = re.escape(sentence)
            # Look for the sentence starting from current position
            search_text: str = text[current_pos:]
            match = re.search(escaped_sentence, search_text)
            
            if match:
                start: int = current_pos + match.start()
                end: int = start + len(sentence)
                offsets.append((start, end))
                current_pos = end  # Move forward to avoid duplicate matches
        
        return offsets

    def _extract_sentence_offsets_with_base(self, text: str, base_offset: int) -> List[Tuple[int, int]]:
        """Extract sentence offsets with adjustment for base offset."""
        offsets: List[Tuple[int, int]] = self._extract_sentence_offsets(text)
        result: List[Tuple[int, int]] = []
        
        for start, end in offsets:
            result.append((base_offset + start, base_offset + end))
            
        return result

    def _split_text_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into chunks while preserving sentence offsets."""
        sentences: List[str] = extract_sentences(text)
        
        splits_with_offsets: List[Tuple[str, int, int]] = []
        current_offset: int = 0

        for sentence in sentences:
            stripped_sentence: str = sentence.strip()
            if not stripped_sentence:
                continue
                
            # Find the exact occurrence of this sentence starting from current_offset
            start: int = text.find(sentence, current_offset)
            if start != -1:
                end: int = start + len(sentence)
                splits_with_offsets.append((sentence, start, end))
                current_offset = end  # Move forward

        # Merge splits into chunks
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
        current_length: int = 0

        for split_item in splits:
            split_text: str = split_item[0]
            start_offset: int = split_item[1]
            end_offset: int = split_item[2]
            split_length: int = len(split_text)
            
            overlap_adjustment: int = self._chunk_overlap if current_chunk else 0
            
            if current_length + split_length <= self._chunk_size - overlap_adjustment:
                current_chunk.append((split_text, start_offset, end_offset))
                current_length += split_length
            else:
                if current_chunk:
                    chunk_parts: List[str] = []
                    for part in current_chunk:
                        chunk_parts.append(part[0])
                    
                    chunk_text: str = "".join(chunk_parts)
                    chunk_start: int = current_chunk[0][1]
                    chunk_end: int = current_chunk[-1][2]
                    merged_splits.append((chunk_text, chunk_start, chunk_end))
                
                current_chunk = [(split_text, start_offset, end_offset)]
                current_length = split_length

        # Add the last chunk
        if current_chunk:
            chunk_parts: List[str] = []
            for part in current_chunk:
                chunk_parts.append(part[0])
            
            chunk_text: str = "".join(chunk_parts)
            chunk_start: int = current_chunk[0][1]
            chunk_end: int = current_chunk[-1][2]
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
        
        for i, chunk_item in enumerate(chunks):
            chunk_text: str = chunk_item[0]
            start_offset: int = chunk_item[1]
            end_offset: int = chunk_item[2]
            
            if i == 0:
                result.append((chunk_text, start_offset, end_offset))
            else:
                # Calculate overlap with previous chunk
                prev_chunk_text: str = chunks[i-1][0]
                overlap_size: int = min(self._chunk_overlap, len(prev_chunk_text))
                overlap_text: str = prev_chunk_text[-overlap_size:]
                
                # Create new chunk with overlap
                new_chunk: str = overlap_text + chunk_text
                # Adjust start offset to account for overlap
                new_start: int = end_offset - len(new_chunk)
                
                result.append((new_chunk, new_start, end_offset))
        
        return result