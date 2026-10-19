import json
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

LABELS = ["neutral", "joy", "sadness", "anger", "enthusiasm", "surprise", "disgust", "fear", "guilt", "shame"]
class EmotionPredictor:
    def __init__(self, artifacts_dir="artifacts"):
        root = Path(artifacts_dir)
        required = [root/"model.keras", root/"tokenizer.json", root/"metadata.json"]
        missing = [str(p) for p in required if not p.exists()]
        if missing: raise FileNotFoundError("Модели не найдены. Сначала выполните python run_experiment.py. Не найдены: " + ", ".join(missing))
        self.model = load_model(root/"model.keras")
        self.tokenizer = tokenizer_from_json((root/"tokenizer.json").read_text(encoding="utf-8"))
        meta = json.loads((root/"metadata.json").read_text(encoding="utf-8"))
        self.max_len = int(meta["max_len"])
        self.thresholds = np.array(meta.get("thresholds", [0.5]*len(LABELS)), dtype=float)
    def predict(self, texts):
        seq = self.tokenizer.texts_to_sequences(texts)
        x = pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")
        probs = self.model.predict(x, verbose=0)
        result=[]
        for text, row in zip(texts, probs):
            chosen = np.flatnonzero(row >= self.thresholds)
            if len(chosen) == 0: chosen = np.array([int(np.argmax(row))])
            result.append({"text":text, "emotions":[LABELS[int(i)] for i in chosen], "scores":{LABELS[i]:float(row[i]) for i in range(len(LABELS))}})
        return result
