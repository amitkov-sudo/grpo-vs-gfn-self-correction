"""Shared code for the GRPO against GFlowNet comparison on planted maths errors.

scoring     final answers, rewards, false alarms, intervals and paired tests
data        starts for the fix test, from GSM8K-SC and from script-edited GSM8K solutions
losses      the five training conditions in PyTorch (CUDA and Tinker backends)
mlx_losses  the same losses in MLX (free Mac backend)
"""
