from pydantic import BaseModel


class Segment(BaseModel):
    start: float
    end: float
    label: str


class AnalysisResult(BaseModel):
    segments: list[Segment]
    duration: float
