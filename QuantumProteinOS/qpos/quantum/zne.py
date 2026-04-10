def zero_noise_extrapolate(circuit_fn, scale_factors=[1, 2, 3]) -> float:
    """
    Richardson extrapolation to zero noise from scaled noise executions.
    """
    results = [circuit_fn(scale) for scale in scale_factors]
    
    # Simple linear fallback / Stub logic
    if len(results) >= 2:
        extrapolated = 2 * results[0] - results[1]
    else:
        extrapolated = results[0] if results else 0.0
        
    return extrapolated
