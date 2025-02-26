"""
Utility functions for scoring sentences based on query relevance.
"""

from typing import Dict, List, Tuple, Any, cast

# Import TChunk and TCitation directly
from fast_graphrag._types import TChunk, TCitation

# Scoring weights
TERM_OVERLAP_WEIGHT = 0.6
POSITION_WEIGHT = 0.3
LENGTH_WEIGHT = 0.1

def score_sentences_in_chunk(chunk: Any, query: str) -> None:
    """Score sentences within a chunk based on relevance to query.
    
    Args:
        chunk: The chunk containing sentences to score
        query: The query to score sentences against
    """
    # Make sure we're dealing with a TChunk
    tchunk = cast(TChunk, chunk)
    
    # Skip if chunk has no citation info
    if tchunk.citation is None:
        return
    
    citation: TCitation = tchunk.citation
    if not citation.sentence_offsets:
        return
    
    # Simple scoring based on term overlap, position, and length
    query_terms = set(query.lower().split())
    scores: Dict[int, float] = {}
    
    for i, (start, end) in enumerate(citation.sentence_offsets):
        # Adjust offsets to be relative to chunk content
        rel_start = max(0, start - citation.start_offset)
        rel_end = min(len(tchunk.content), end - citation.start_offset)
        
        if rel_start < rel_end and rel_end <= len(tchunk.content):
            sentence = tchunk.content[rel_start:rel_end].strip()
            if not sentence:  # Skip empty sentences
                continue
                
            sentence_terms = set(sentence.lower().split())
            
            # Calculate term overlap score
            if query_terms and sentence_terms:
                overlap = len(query_terms.intersection(sentence_terms))
                term_overlap_score = overlap / len(query_terms)
            else:
                term_overlap_score = 0.0
            
            # Calculate position score (earlier sentences get higher scores)
            position_score = 1.0 - (i / max(1, len(citation.sentence_offsets)))
            
            # Calculate length normalization (prefer medium-length sentences)
            optimal_length = 100  # characters
            length_score = 1.0 - min(1.0, abs(len(sentence) - optimal_length) / optimal_length)
            
            # Combine scores with weights
            scores[i] = (
                TERM_OVERLAP_WEIGHT * term_overlap_score +
                POSITION_WEIGHT * position_score +
                LENGTH_WEIGHT * length_score
            )
    
    # Update the chunk with sentence scores
    if scores:
        citation.sentence_scores = scores

def get_highlighted_sentences(chunks: List[Tuple[Any, float]], query: str) -> List[str]:
    """Extract and highlight the most relevant sentences from chunks.
    
    Args:
        chunks: List of (chunk, score) tuples
        query: The query used for searching
        
    Returns:
        List of formatted sentences with source information
    """
    results: List[str] = []
    
    for chunk, _ in chunks:
        tchunk = cast(TChunk, chunk)
        score_sentences_in_chunk(tchunk, query)
        
        if tchunk.citation and tchunk.citation.sentence_scores:
            best_sentence = tchunk.citation.get_most_relevant_sentence(tchunk.content)
            if best_sentence:
                # Add to results with reference information
                doc_title = tchunk.metadata.get('title', 'document')
                results.append(f"• {best_sentence.strip()} [Source: {doc_title}]")
            else:
                # Fallback to a snippet from the chunk
                snippet = tchunk.content[:150] + "..." if len(tchunk.content) > 150 else tchunk.content
                results.append(f"• {snippet.strip()} [Source: {tchunk.metadata.get('title', 'document')}]")
        else:
            # Fallback for chunks without enhanced citation
            snippet = tchunk.content[:150] + "..." if len(tchunk.content) > 150 else tchunk.content
            results.append(f"• {snippet.strip()} [Source: {tchunk.metadata.get('title', 'document')}]")
    
    return results

def extract_relevant_sentences(chunk: Any, query: str, max_sentences: int = 3) -> List[str]:
    """Extract the most relevant sentences from a chunk based on query.
    
    Args:
        chunk: The chunk to extract sentences from
        query: The query to score sentences against
        max_sentences: Maximum number of sentences to extract
        
    Returns:
        List of the most relevant sentences
    """
    tchunk = cast(TChunk, chunk)
    result_sentences: List[str] = []
    
    if tchunk.citation is None:
        # Fall back to returning the whole chunk content or a substring
        fallback = tchunk.content[:200] + "..." if len(tchunk.content) > 200 else tchunk.content
        result_sentences.append(fallback)
        return result_sentences
        
    # Score sentences if not already scored
    score_sentences_in_chunk(tchunk, query)
    
    citation = tchunk.citation
    
    if not citation.sentence_scores:
        fallback = tchunk.content[:200] + "..." if len(tchunk.content) > 200 else tchunk.content
        result_sentences.append(fallback)
        return result_sentences
        
    # Get top scoring sentence indices
    top_indices = sorted(
        citation.sentence_scores.keys(), 
        key=lambda i: citation.sentence_scores.get(i, 0), 
        reverse=True
    )[:max_sentences]
    
    # Extract and add sentences
    for idx in sorted(top_indices):  # Sort by position, not score
        if idx < len(citation.sentence_offsets):
            start, end = citation.sentence_offsets[idx]
            # Adjust offsets relative to chunk content
            rel_start = max(0, start - citation.start_offset)
            rel_end = min(len(tchunk.content), end - citation.start_offset)
            
            if rel_start < rel_end and rel_end <= len(tchunk.content):
                result_sentences.append(tchunk.content[rel_start:rel_end])
    
    return result_sentences