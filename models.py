from pydantic import BaseModel, Field
from typing import Optional, Literal


class Visualization(BaseModel):
    title: str = Field(description="Title of the visualization, describing what it shows")
    type: Literal["chart", "mermaid"] = Field(description="Type of visualization: 'chart' for data charts, 'mermaid' for diagram code")
    chart_type: Optional[Literal["bar", "line", "pie", "scatter", "area"]] = Field(
        default=None,
        description="Chart subtype when type='chart'. One of: bar, line, pie, scatter, area"
    )
    labels: Optional[list[str]] = Field(
        default=None,
        description="X-axis labels or category names for chart visualizations"
    )
    values: Optional[list[float]] = Field(
        default=None,
        description="Numeric values corresponding to labels for chart visualizations"
    )
    code: Optional[str] = Field(
        default=None,
        description="Mermaid diagram code string when type='mermaid'"
    )
    description: str = Field(description="Human-readable explanation of what insight this visualization conveys")


class AnalysisOutput(BaseModel):
    key_themes: list[str] = Field(description="List of 3-7 key themes identified in the research data")
    summary: str = Field(description="Concise 2-4 sentence summary of the overall research findings")
    visualizations: list[Visualization] = Field(description="List of 2-4 visualizations representing the most important data insights")


class QCOutput(BaseModel):
    score: int = Field(ge=0, le=100, description="Overall quality score from 0 to 100")
    passed: bool = Field(description="True if score >= 85, indicating research meets quality threshold")
    feedback: str = Field(description="Actionable feedback for the researcher if the score is below threshold, or a summary of strengths if passing")
    strengths: list[str] = Field(description="List of specific strengths observed in the research data")
    weaknesses: list[str] = Field(description="List of specific weaknesses or gaps that should be addressed in a retry")
