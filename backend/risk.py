"""
risk.py
-------
Turns the model's "probability that this voice is synthetic"
into a simple 0-100 risk score and a human-friendly risk level.

Kept in its own file/function on purpose, so the scoring logic
can be tuned later without touching the ML or API code.

NOTE: This is an application-level risk indicator for a hackathon
demo. It is NOT a clinically or security-validated score.
"""


def calculate_risk_score(synthetic_probability: float) -> int:
    """
    Converts a probability (0.0 - 1.0) that the voice is synthetic
    into a risk score from 0 to 100.

    Example: probability 0.874 -> risk score 87
    """
    # Clamp probability to a safe 0-1 range just in case
    probability = max(0.0, min(1.0, synthetic_probability))
    risk_score = round(probability * 100)
    return risk_score


def get_risk_level(risk_score: int) -> str:
    """
    Buckets a 0-100 risk score into LOW / MEDIUM / HIGH.
    """
    if risk_score <= 30:
        return "LOW"
    elif risk_score <= 60:
        return "MEDIUM"
    else:
        return "HIGH"


def get_security_message(risk_level: str) -> str:
    """
    Returns the user-facing guidance shown for each risk level.
    """
    if risk_level == "HIGH":
        return (
            "Potentially suspicious voice detected. Verify the caller "
            "through another trusted method before sharing sensitive "
            "information."
        )
    elif risk_level == "MEDIUM":
        return (
            "This voice shows some characteristics that warrant caution. "
            "Consider verifying the caller through another trusted method."
        )
    else:
        return (
            "Voice appears less suspicious based on this analysis. "
            "Continue to use normal security precautions. This result "
            "does not prove the caller is genuine."
        )
