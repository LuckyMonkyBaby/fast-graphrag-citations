# type: ignore
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import xxhash

from fast_graphrag._services._chunk_extraction import DefaultChunkingService, find_sentence_boundaries
from fast_graphrag._types import THash, TCitation


@dataclass
class MockDocument:
    data: str
    metadata: Dict[str, Any]


@dataclass
class MockChunk:
    id: THash
    content: str
    metadata: Dict[str, Any]
    citation: TCitation


class TestDefaultChunkingService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.chunking_service = DefaultChunkingService()

    async def test_extract(self):
        doc1 = MockDocument(data="test data 1.", metadata={"meta": "data1"})
        doc2 = MockDocument(data="test data 2.", metadata={"meta": "data2"})
        documents = [doc1, doc2]

        # Create mock citations with sentence offsets
        citation1 = TCitation(
            start_offset=0,
            end_offset=len(doc1.data),
            sentence_offsets=[(0, len(doc1.data))]
        )
        
        with patch.object(
            self.chunking_service,
            "_extract_chunks",
            return_value=[
                MockChunk(
                    id=THash(xxhash.xxh3_64_intdigest(doc1.data)), 
                    content=doc1.data, 
                    metadata=doc1.metadata,
                    citation=citation1
                )
            ],
        ) as mock_extract_chunks:
            chunks = await self.chunking_service.extract(documents)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 1)
        self.assertEqual(chunks[0][0].content, "test data 1.")
        self.assertEqual(chunks[0][0].metadata, {"meta": "data1"})
        self.assertEqual(chunks[0][0].citation.start_offset, 0)
        self.assertEqual(chunks[0][0].citation.end_offset, len(doc1.data))
        self.assertEqual(chunks[0][0].citation.sentence_offsets, [(0, len(doc1.data))])
        mock_extract_chunks.assert_called()

    async def test_extract_with_duplicates(self):
        doc1 = MockDocument(data="test data 1.", metadata={"meta": "data1"})
        doc2 = MockDocument(data="test data 1.", metadata={"meta": "data1"})
        documents = [doc1, doc2]

        # Create mock citations with sentence offsets
        citation1 = TCitation(
            start_offset=0,
            end_offset=len(doc1.data),
            sentence_offsets=[(0, len(doc1.data))]
        )
        
        with patch.object(
            self.chunking_service,
            "_extract_chunks",
            return_value=[
                MockChunk(
                    id=THash(xxhash.xxh3_64_intdigest(doc1.data)), 
                    content=doc1.data, 
                    metadata=doc1.metadata,
                    citation=citation1
                )
            ],
        ) as mock_extract_chunks:
            chunks = await self.chunking_service.extract(documents)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 1)
        self.assertEqual(len(chunks[1]), 1)
        self.assertEqual(chunks[0][0].content, "test data 1.")
        self.assertEqual(chunks[0][0].metadata, {"meta": "data1"})
        self.assertEqual(chunks[0][0].citation.sentence_offsets, [(0, len(doc1.data))])
        self.assertEqual(chunks[1][0].content, "test data 1.")
        self.assertEqual(chunks[1][0].metadata, {"meta": "data1"})
        self.assertEqual(chunks[1][0].citation.sentence_offsets, [(0, len(doc1.data))])
        mock_extract_chunks.assert_called()

    async def test_extract_chunks(self):
        doc = MockDocument(data="test data.", metadata={"meta": "data"})
        
        chunks = await self.chunking_service._extract_chunks(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, doc.data)
        self.assertEqual(chunks[0].metadata, doc.metadata)
        self.assertEqual(chunks[0].citation.start_offset, 0)
        self.assertEqual(chunks[0].citation.end_offset, len(doc.data))
        self.assertEqual(chunks[0].citation.sentence_offsets, [(0, len(doc.data))])

    def test_find_sentence_boundaries(self):
        # Test with simple sentences
        text = "This is a sentence. This is another sentence. And a third one!"
        boundaries = find_sentence_boundaries(text)
        self.assertEqual(boundaries, [(0, 19), (19, 44), (44, 60)])
        
        # Test with Chinese punctuation
        text = "这是一个句子。这是另一个句子！最后一个句子？"
        boundaries = find_sentence_boundaries(text)
        self.assertEqual(boundaries, [(0, 7), (7, 14), (14, 21)])
        
        # Test with no sentence boundaries
        text = "This is text without sentence boundaries"
        boundaries = find_sentence_boundaries(text)
        self.assertEqual(boundaries, [(0, len(text))])
        
        # Test with empty text
        text = ""
        boundaries = find_sentence_boundaries(text)
        self.assertEqual(boundaries, [])

    async def test_extract_chunks_with_multiple_sentences(self):
        doc = MockDocument(
            data="First sentence. Second sentence. Third sentence.",
            metadata={"meta": "data"}
        )
        
        chunks = await self.chunking_service._extract_chunks(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, doc.data)
        self.assertEqual(chunks[0].metadata, doc.metadata)
        self.assertEqual(chunks[0].citation.start_offset, 0)
        self.assertEqual(chunks[0].citation.end_offset, len(doc.data))
        
        # Check sentence offsets
        expected_offsets = [(0, 15), (15, 32), (32, 48)]
        self.assertEqual(chunks[0].citation.sentence_offsets, expected_offsets)

    async def test_get_sentence_offsets_with_base(self):
        text = "First sentence. Second sentence."
        base_offset = 10
        
        offsets = self.chunking_service._get_sentence_offsets_with_base(text, base_offset)
        expected_offsets = [(10, 25), (25, 42)]
        self.assertEqual(offsets, expected_offsets)


if __name__ == "__main__":
    unittest.main()