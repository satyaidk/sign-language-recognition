"""
L3 — inference: one shared predictor for file and live recognition.

    predictor  load the exported model; landmarks -> sign (+ robust window voting)
    segmenter  motion-gated sign segmentation + repeat de-duplication
    video      stage 4: recognise a video file, render a prediction overlay
    live       stage 5: real-time webcam recognition (streaming landmarks)
"""
