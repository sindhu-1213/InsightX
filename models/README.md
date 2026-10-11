# Models Directory

Per workspace rules:
- Model weights or feature arrays over 25 MB (`*.pkl`, `*.npy`, `*.h5`, `*.onnx`) are gitignored.
- Only lightweight metadata configuration files (like `model_config.json`) are committed.
- Large binary artifacts must be generated locally via `insightx.train` or downloaded from release storage.
