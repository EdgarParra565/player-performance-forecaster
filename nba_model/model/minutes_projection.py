"""Minutes projection helpers."""


def project_minutes(
    avg_minutes: float,
    spread: float,
    blowout_threshold: float = 10.0,
    blowout_penalty: float = 0.12
) -> float:
    """
    Project expected minutes from baseline minutes and spread context.

    Args:
        avg_minutes: Baseline expected minutes.
        spread: Absolute spread magnitude (e.g., -12 -> 12).
        blowout_threshold: Spread threshold to treat game as potential blowout.
        blowout_penalty: Fractional minutes reduction when blowout threshold is hit.

    Returns:
        Projected minutes after blowout-risk adjustment.
    """
    if spread >= blowout_threshold:
        return avg_minutes * (1 - blowout_penalty)

    return avg_minutes
