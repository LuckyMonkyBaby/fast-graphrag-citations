"""Prompts."""

from typing import Any, Dict

PROMPTS: Dict[str, Any] = {}

## NEW
PROMPTS["entity_relationship_extraction"] = """# DOMAIN PROMPT
{domain}

# GOAL
Your goal is to highlight information that is relevant to the domain and the questions that may be asked on it.
Given an input document, identify all relevant entities and all relationships among them.

Examples of possible questions:
{example_queries}

# STEPS
1. Identify all entities of the given types. Make sure to extract all and only the entities that are of one of the given types. Use singular names and split compound concepts when necessary (for example, from the sentence "they are movie and theater directors", you should extract the entities "movie director" and "theater director").
2. Identify all relationships between the entities found in step 1. Clearly resolve pronouns to their specific names to maintain clarity.
3. Double check that each entity identified in step 1 appears in at least one relationship. If not, add the missing relationships.

# EXAMPLE DATA
Example types: [location, organization, person, communication]
Example document: Radio City: Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into new media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."

Output:
{{
"entities": [
	{{"name": "RADIO CITY", "type": "organization", "desc": "Radio City is India's first private FM radio station"}},
	{{"name": "INDIA", "type": "location", "desc": "A country"}},
	{{"name": "FM RADIO STATION", "type": "communication", "desc": "A radio station that broadcasts using frequency modulation"}},
	{{"name": "ENGLISH", "type": "communication", "desc": "A language"}},
	{{"name": "HINDI", "type": "communication", "desc": "A language"}},
	{{"name": "NEW MEDIA", "type": "communication", "desc": "New media"}},
	{{"name": "PLANETRADIOCITY", "type": "organization", "desc": "PlanetRadiocity.com is an online music portal"}},
	{{"name": "MUSIC PORTAL", "type": "communication", "desc": "A website that offers music related information"}},
	{{"name": "NEWS", "type": "communication", "desc": "News"}},
	{{"name": "VIDEO", "type": "communication", "desc": "Video"}},
	{{"name": "SONG", "type": "communication", "desc": "Song"}}
],
"relationships": [
	{{"source": "RADIO CITY", "target": "INDIA", "desc": "Radio City is located in India"}},
	{{"source": "RADIO CITY", "target": "FM RADIO STATION", "desc": "Radio City is a private FM radio station started on 3 July 2001"}},
	{{"source": "RADIO CITY", "target": "ENGLISH", "desc": "Radio City broadcasts English songs"}},
	{{"source": "RADIO CITY", "target": "HINDI", "desc": "Radio City broadcasts songs in the Hindi language"}},
	{{"source": "RADIO CITY", "target": "PLANETRADIOCITY", "desc": "Radio City launched PlanetRadiocity.com in May 2008"}},
	{{"source": "PLANETRADIOCITY", "target": "MUSIC PORTAL", "desc": "PlanetRadiocity.com is a music portal"}},
	{{"source": "PLANETRADIOCITY", "target": "NEWS", "desc": "PlanetRadiocity.com offers music related news"}},
	{{"source": "PLANETRADIOCITY", "target": "SONG", "desc": "PlanetRadiocity.com offers songs"}}
],
"other_relationships": [
	{{"source": "RADIO CITY", "target": "NEW MEDIA", "desc": "Radio City forayed into new media in May 2008."}},
	{{"source": "PLANETRADIOCITY", "target": "VIDEO", "desc": "PlanetRadiocity.com offers music related videos"}}
]
}}

# INPUT DATA
Types: {entity_types}
Document: {input_text}

Output:
"""

PROMPTS["entity_relationship_continue_extraction"] = "MANY entities were missed in the last extraction.  Add them below using the same format:"

PROMPTS["entity_relationship_gleaning_done_extraction"] = "Retrospectively check if all entities have been correctly identified: answer done if so, or continue if there are still entities that need to be added."

PROMPTS["entity_extraction_query"] = """Given the query below, your task is to extract all entities relevant to perform information retrieval to produce an answer.

-EXAMPLE 1-
Query: Who directed the film that was shot in or around Leland, North Carolina in 1986?
Ouput: {{"named": ["[PLACE] Leland", "[COUNTRY] North Carolina", "[YEAR] 1986"], "generic": ["film director"]}}

-EXAMPLE 2-
Query: What relationship does Fred Gehrke have to the 23rd overall pick in the 2010 Major League Baseball Draft?
Ouput: {{"named": ["[BASEBALL PLAYER] Fred Gehrke", "[EVENT] 2010 Major League Baseball Draft"], "generic": ["23rd baseball draft pick"]}}

-INPUT-
Query: {query}
Output:
"""


PROMPTS[
	"summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a summary of the data provided below.
Given the current description, summarize it by removing redundant and generic information. Resolve any contradictions and provide a single, coherent summary.
Write in third person and explicitly include the entity names to preserve the full context.

Current:
{description}

Updated:
"""


PROMPTS[
	"edges_group_similar"
] = """You are a helpful assistant responsible for maintaining a list of facts describing the relations between two entities so that information is not redundant.
Given a list of ids and facts, identify any facts that should be grouped together as they contain similar or duplicated information and provide a new summarized description for the group.

# EXAMPLE
Facts (id, description):
0, Mark is the dad of Luke
1, Luke loves Mark
2, Mark is always ready to help Luke
3, Mark is the father of Luke
4, Mark loves Luke very much

Ouput:
{{
	grouped_facts: [
	{{
		'ids': [0, 3],
		'description': 'Mark is the father of Luke'
	}},
	{{
		'ids': [1, 4],
		'description': 'Mark and Luke love each other very much'
	}}
	]
}}

# INPUT:
Facts:
{edge_list}

Ouput:
"""

PROMPTS["generate_response_query_with_references"] = """You are a helpful assistant analyzing the given input data to provide an helpful response to the user query.

# INPUT DATA
{context}

# USER QUERY
{query}

# INSTRUCTIONS
Your goal is to provide a response to the user query using the relevant information in the input data:
- the "Entities" and "Relationships" tables contain high-level information. Use these tables to identify the most important entities and relationships to respond to the query.
- the "Sources" list contains raw text sources to help answer the query. It may contain noisy data, so pay attention when analyzing it.

Follow these steps:
1. Read and understand the user query.
2. Look at the "Entities" and "Relationships" tables to get a general sense of the data and understand which information is the most relevant to answer the query.
3. Carefully analyze all the "Sources" to get more detailed information. Information could be scattered across several sources, use the identified relevant entities and relationships to guide yourself through the analysis of the sources.
4. While you write the response, you must include inline references to the all the sources you are using by appending `[<source_id>]` at the end of each sentence, where `source_id` is the corresponding source ID from the "Sources" list.
5. Write the response to the user query - which must include the inline references - based on the information you have gathered. Be very concise and answer the user query directly. If the response cannot be inferred from the input data, just say no relevant information was found. Do not make anything up or add unrelevant information.

Answer:
"""

PROMPTS["generate_response_query_no_references"] = """You are a helpful assistant analyzing the given input data to provide an helpful response to the user query.

# INPUT DATA
{context}

# USER QUERY
{query}

# INSTRUCTIONS
Your goal is to provide a response to the user query using the relevant information in the input data:
- the "Entities" and "Relationships" tables contain high-level information. Use these tables to identify the most important entities and relationships to respond to the query.
- the "Sources" list contains raw text sources to help answer the query. It may contain noisy data, so pay attention when analyzing it.

Follow these steps:
1. Read and understand the user query.
2. Look at the "Entities" and "Relationships" tables to get a general sense of the data and understand which information is the most relevant to answer the query.
3. Carefully analyze all the "Sources" to get more detailed information. Information could be scattered across several sources, use the identified relevant entities and relationships to guide yourself through the analysis of the sources.
4. Write the response to the user query based on the information you have gathered. Be very concise and answer the user query directly. If the response cannot be inferred from the input data, just say no relevant information was found. Do not make anything up or add unrelevant information.

Answer:
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."


PROMPTS["generate_json_response_query"] = """You are a precise assistant analyzing input data to provide concise JSON responses to user queries.

# INPUT DATA
{context}

# USER QUERY
{query}

# INSTRUCTIONS
Your goal is to provide a short, structured JSON response to the user query using only relevant information from the input data:
- The "Entities" and "Relationships" tables contain high-level information
- The "Sources" list contains raw text sources with detailed information
- Your output must be valid JSON with the exact structure shown below

Follow these steps:
1. Identify the core information needed to answer the query
2. Extract only the most relevant facts from the sources
3. Format your response as a concise JSON object with these fields:
   - "answer": A short factual response (1-2 sentences maximum)
   - "confidence": A number between 0-1 indicating your confidence level
   - "entities": Array of key entity names referenced in your answer
   - "source_ids": Array of source IDs that support your answer

Output format:
{
  "answer": "Your concise factual response here",
  "confidence": 0.95,
  "entities": ["ENTITY1", "ENTITY2"],
  "source_ids": ["source_1", "source_2"]
}

If no relevant information is found, return:
{
  "answer": "No relevant information found",
  "confidence": 0,
  "entities": [],
  "source_ids": []
}

Response (valid JSON only):
"""

# Schema-based JSON response prompt
PROMPTS["generate_schema_json_response_query"] = """You are a precise assistant analyzing input data to provide schema-compliant JSON responses to user queries.

# INPUT DATA
{context}

# USER QUERY
{query}

# INSTRUCTIONS
Your goal is to provide a structured JSON response to the user query using only relevant information from the input data:
- The "Entities" and "Relationships" tables contain high-level information
- The "Sources" list contains raw text sources with detailed information
- Your output must be valid JSON conforming exactly to the schema specification below

JSON SCHEMA:
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "The original user query"
    },
    "response": {
      "type": "object",
      "properties": {
        "main_answer": {
          "type": "string",
          "description": "Direct factual answer to the query (1-3 sentences)"
        },
        "key_entities": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": {"type": "string"},
              "type": {"type": "string"},
              "relevance": {"type": "number", "minimum": 0, "maximum": 1}
            }
          }
        },
        "key_relationships": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "source": {"type": "string"},
              "target": {"type": "string"},
              "description": {"type": "string"}
            }
          }
        },
        "supporting_evidence": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "source_id": {"type": "string"},
              "content_snippet": {"type": "string"},
              "relevance": {"type": "number", "minimum": 0, "maximum": 1}
            }
          }
        },
        "confidence": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "Overall confidence in the answer"
        }
      },
      "required": ["main_answer", "confidence"]
    }
  },
  "required": ["query", "response"]
}

Follow these steps:
1. Identify the core information needed to answer the query
2. Extract only the most relevant facts from the sources
3. Format your response as a valid JSON object that conforms exactly to the schema
4. Ensure all required fields are present
5. If no relevant information is found, set confidence to 0 and indicate this in main_answer

Response (valid JSON only):
"""

# Table format response prompt
PROMPTS["generate_table_response_query"] = """You are a helpful assistant analyzing input data to provide tabular responses to user queries.

# INPUT DATA
{context}

# USER QUERY
{query}

# INSTRUCTIONS
Your goal is to provide a tabular response to the user query using relevant information from the input data:
- The "Entities" and "Relationships" tables contain high-level information
- The "Sources" list contains raw text sources with detailed information

Follow these steps:
1. Assess whether the query is best answered with a table format
2. Identify the key columns that would structure the answer effectively
3. Extract relevant information from sources to populate the table
4. Format your response as a markdown table with appropriate headers
5. Include a brief 1-2 sentence introduction before the table
6. Add a short conclusion after the table if necessary

For example:

Here is a summary of the key climate factors affecting sea levels:

| Factor | Impact Level | Mechanism | Time Scale |
|--------|--------------|-----------|------------|
| Thermal Expansion | High | Ocean water expands as it warms | Decades |
| Glacier Melting | Medium | Mountain glaciers add water to oceans | Years to decades |
| Ice Sheet Loss | Very High | Greenland and Antarctic ice sheets melt | Decades to centuries |
| Land Water Storage | Low | Changes in terrestrial water storage | Variable |

These factors combine to create the current rate of sea level rise of approximately 3.6mm per year.

If a table format is not appropriate for the query, provide a concise paragraph response instead.

Response:
"""

# Bullet point response prompt
PROMPTS["generate_bullet_points_response_query"] = """You are a helpful assistant analyzing input data to provide organized bullet point responses to user queries.

# INPUT DATA
{context}

# USER QUERY
{query}

# INSTRUCTIONS
Your goal is to provide a clear, structured bullet point response to the user query using relevant information from the input data:
- The "Entities" and "Relationships" tables contain high-level information
- The "Sources" list contains raw text sources with detailed information

Follow these steps:
1. Identify the key points that address the user's query
2. Organize these points in a logical hierarchy (main points and sub-points)
3. Format your response as bullet points using markdown syntax
4. Begin with a brief 1-2 sentence introduction
5. Group related points under clear headings if helpful
6. End with a short conclusion if appropriate

Example format:

Based on the available information, here are the key factors affecting ocean acidification:

**Primary Causes:**
* Increased atmospheric CO2 levels
  * Human CO2 emissions from fossil fuels
  * Deforestation reducing CO2 absorption
* Changes in ocean circulation patterns
  * Warming-induced stratification
  * Altered upwelling patterns

**Major Impacts:**
* Reduced calcification in marine organisms
* Ecosystem disruption
* Food web alterations

The most significant factor is atmospheric CO2 absorption, accounting for approximately 80% of ocean acidification.

Response:
"""