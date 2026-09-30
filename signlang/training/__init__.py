"""
L2 — learning pipeline: features, augmentation, model, k-fold CV, final fit, ONNX export.

    features   raw (T, 1692) -> fixed (L, D) model input (shared by training AND inference)
    data       dataset index, geometry cache, augmentation, torch Dataset
    model      small BiGRU / Transformer classifier
    train      stage 1: stratified k-fold cross-validation (honest accuracy)
    finetune   stage 2: fit the deployment model on all data (EMA weights)
    export     stage 3: ONNX export + parity check + latency
    viz        training plots
"""
