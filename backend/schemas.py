from dataclasses import dataclass
from pydantic import BaseModel, Field

@dataclass
class RetrievedChunk:
    chunk_id: int
    score: float
    page_number: int
    source_doc: str
    text: str

class Citation(BaseModel):
    chunk_id: int = Field(description="ID of the chunk this citation references")
    quote: str = Field(description="The exact span from the chunk that supports the claim")
    verified: bool = Field(default=False, description="True if the quote was found verbatim in the source chunk")


class Response(BaseModel):
    answer: str = Field(description="The synthesized answer to the user's question")
    citations: list[Citation] = Field(description="One citation per claim in the answer")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-reported confidence, 0.0–1.0")