from pydantic import BaseModel, Field


class PoliticalCompassSubset(BaseModel):
    data_path: str = Field(..., description="Path to the sub-dataset file")


class MoralFoundationSubset(BaseModel):
    data_path: str = Field(..., description="Path to the sub-dataset file")


class TechnologyAiSubset(BaseModel):
    data_path: str = Field(..., description="Path to the sub-dataset file")


class FinancialRiskSubset(BaseModel):
    data_path: str = Field(..., description="Path to the sub-dataset file")


class GeneratedDataset(BaseModel):
    political_compass: PoliticalCompassSubset = Field(
        ..., description="Configuration for the political compass subset"
    )
    moral_foundations: MoralFoundationSubset = Field(
        ..., description="Configuration for the moral foundation subset"
    )
    technology_ai: TechnologyAiSubset = Field(
        ..., description="Configuration for the technology and AI subset"
    )
    financial_risk: FinancialRiskSubset = Field(
        ..., description="Configuration for the financial risk subset"
    )

    model_used: str = Field(
        "gemini/gemini-2.5-pro", description="The model used for dataset generation"
    )


class Config(BaseModel):
    generated_dataset: GeneratedDataset = Field(
        ..., description="Configuration for generated datasets"
    )
