from dataclasses import dataclass, field
from typing import List, Optional

# result schema for guardrail validation
@dataclass
class GuardrailResult:
    passed: bool
    cleaned_text: str
    violations: List[str] = field(default_factory=list)
    reason: Optional[str] = None
