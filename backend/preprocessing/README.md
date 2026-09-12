# Preprocessing

`pipeline.py` performs configurable quality checks, a conservative non-background crop, LAB-space CLAHE, resize, and ImageNet normalization. The same module is used by inference. Thresholds can be changed with `MIN_BRIGHTNESS`, `MAX_BRIGHTNESS`, `MIN_BLUR_SCORE`, and `MIN_RETINAL_AREA`.

The output metadata is intended for visualization and reproducibility; preprocessing does not establish clinical validity.
