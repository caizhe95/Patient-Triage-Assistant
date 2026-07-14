 
from pydantic import BaseModel, Field, field_validator


ALLOWED_TRIAGE_DOMAINS = {
    "general",
    "respiratory",
    "gastroenterology",
    "dermatology",
    "orthopedics",
    "gynecology",
    "pediatrics",
    "neurology",
    "cardiology",
}


class TriagePlan(BaseModel):
    domain: str
    candidate_domains: list[str] = Field(default_factory=list)
    retrieval_query: str
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("domain")
    @classmethod
    def domain_must_be_supported(cls, value: str) -> str:
        if value not in ALLOWED_TRIAGE_DOMAINS:
            raise ValueError(f"unsupported triage domain: {value}")
        return value

    @field_validator("candidate_domains")
    @classmethod
    def candidate_domains_must_be_supported(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in ALLOWED_TRIAGE_DOMAINS]
        if invalid:
            raise ValueError(f"unsupported candidate domains: {invalid}")
        return list(dict.fromkeys(values))
