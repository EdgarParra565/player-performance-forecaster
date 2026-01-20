def project_minutes(
    avg_minutes: float,
    spread: float,
    blowout_threshold: float = 10.0,
    blowout_penalty: float = 0.12
) -> float:
    """
    Projects expected minutes based on Vegas spread

    spread: absolute value (e.g. -12 → 12)
    """
    if spread >= blowout_threshold:
        return avg_minutes * (1 - blowout_penalty)

    return avg_minutes
