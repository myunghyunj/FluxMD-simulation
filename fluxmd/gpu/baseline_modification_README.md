# FluxMD GPU Implementation Notes

## Temporal Fluctuation (τᵢ) Calculation

The flux formula **Φᵢ = ⟨|E̅ᵢ|⟩ · Cᵢ · (1 + τᵢ)** uses different methods for calculating τᵢ:

### CPU Version
```python
# Uses numerical derivative of force magnitude
mag_derivative = savgol_filter(magnitudes, window_length=5, polyorder=2, deriv=1)
rate_of_change = np.sqrt(np.mean(mag_derivative**2))  # This is τᵢ
```

### GPU Version
```python
# Uses normalized variance of interaction energies
variance = squared_deviations / interaction_count
temporal_factor = 1.0 + torch.sqrt(variance) / (avg_energy + 1e-10)  # This is (1 + τᵢ)
```

## Key Differences

1. **Mathematical Approach**:
   - CPU: Time derivative → rate of change
   - GPU: Statistical variance → energy fluctuation

2. **Implementation**:
   - CPU: Sequential processing with NumPy
   - GPU: Parallel scatter operations with PyTorch

3. **Performance**:
   - CPU: Better for small systems or debugging
   - GPU: Optimized for large-scale parallel computation

Both methods capture temporal dynamics but use different mathematical representations suitable for their respective architectures.