import json, os, random, re
from pathlib import Path
import numpy as np
import pandas as pd
import keras_tuner as kt
from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .model import LABELS

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / "data", ROOT / "artifacts"
MAX_WORDS, MAX_LEN, VECTOR_SIZE, SEED = 5000, 100, 100, 42
TRIALS = int(os.getenv("EMOSCOPE_MAX_TRIALS", "5"))
EPOCHS = int(os.getenv("EMOSCOPE_EPOCHS", "5"))
BATCH = int(os.getenv("EMOSCOPE_BATCH_SIZE", "64"))


def preprocess(text):
    text = str(text).lower()
    try:
        from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
        if not hasattr(preprocess, "pipeline"):
            preprocess.pipeline = (Segmenter(), MorphVocab(), NewsMorphTagger(NewsEmbedding()))
        segmenter, vocab, tagger = preprocess.pipeline
        doc = Doc(text); doc.segment(segmenter); doc.tag_morph(tagger)
        tokens = []
        for token in doc.tokens:
            token.lemmatize(vocab)
            if token.text.isalnum(): tokens.append(token.lemma or token.text)
        return " ".join(tokens)
    except Exception:
        return " ".join(re.findall(r"[а-яёa-z0-9]+", text))


def load_data():
    train_path, test_path = DATA / "data_train.csv", DATA / "data_test.csv"
    if train_path.exists() and test_path.exists():
        print("Используются локальные CSV-файлы из data/.")
        return pd.read_csv(train_path), pd.read_csv(test_path)
    print("Загружаю Djacon/ru-izard-emotions с Hugging Face...")
    ds = load_dataset("Djacon/ru-izard-emotions")
    def frame(split):
        df = ds[split].to_pandas()
        text_col = next((c for c in ("text", "text_comment", "comment", "sentence") if c in df), None)
        if text_col is None: raise ValueError(f"Не найдена колонка текста: {list(df.columns)}")
        missing = set(LABELS) - set(df.columns)
        if missing: raise ValueError(f"Не найдены метки {sorted(missing)}; колонки: {list(df.columns)}")
        return df[[text_col, *LABELS]].rename(columns={text_col: "text"})
    splits = list(ds.keys()); train_split = "train" if "train" in splits else splits[0]
    test_split = "test" if "test" in splits else ("validation" if "validation" in splits else None)
    if not test_split: raise ValueError(f"Нет test/validation split: {splits}")
    train, test = frame(train_split), frame(test_split)
    DATA.mkdir(parents=True, exist_ok=True); train.to_csv(train_path, index=False); test.to_csv(test_path, index=False)
    return train, test


def clean_frames(train, test):
    for df in (train, test):
        missing = set(LABELS) - set(df.columns)
        if missing: raise ValueError(f"Отсутствуют эмоциональные метки: {sorted(missing)}")
        text_col = "text" if "text" in df.columns else df.columns[0]
        df["text"] = df[text_col].fillna("").astype(str)
        df["processed_text"] = df["text"].map(preprocess)
        df[LABELS] = df[LABELS].fillna(0).astype(int)
    train = train[train[LABELS].sum(axis=1) > 0].reset_index(drop=True)
    test = test[test[LABELS].sum(axis=1) > 0].reset_index(drop=True)
    if train.empty or test.empty: raise ValueError("Пустой train/test после фильтрации примеров без меток")
    return train, test


def build_vectorizers(texts):
    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    vocab = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    sentences = [s.split() for s in texts if s.strip()] or [["пустой"]]
    w2v = Word2Vec(sentences=sentences, vector_size=VECTOR_SIZE, window=5, min_count=1, workers=1, seed=SEED)
    rng = np.random.default_rng(SEED)
    matrices = {}
    for name in ("word2vec", "random_embedding"):
        mat = rng.normal(0, .05, (vocab, VECTOR_SIZE)).astype("float32"); mat[0] = 0
        if name == "word2vec":
            for word, idx in tokenizer.word_index.items():
                if idx < vocab and word in w2v.wv: mat[idx] = w2v.wv[word]
        matrices[name] = mat
    return tokenizer, matrices


def encode(texts, tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_LEN, padding="post", truncating="post")


def build_model(hp, matrix, output_dim):
    model = keras.Sequential(name="emotion_model")
    model.add(layers.Embedding(matrix.shape[0], matrix.shape[1], weights=[matrix], input_length=MAX_LEN, trainable=True))
    model.add(layers.SpatialDropout1D(hp.Float("spatial_dropout", 0, .5, step=.1)))
    units = hp.Int("units", 50, 150, step=20)
    kwargs = dict(units=units, dropout=hp.Float("dropout", 0, .5, step=.1), recurrent_dropout=hp.Float("recurrent_dropout", 0, .5, step=.1))
    cell = layers.LSTM(**kwargs)
    model.add(cell if hp.Choice("architecture", ["LSTM", "BiLSTM"]) == "LSTM" else layers.Bidirectional(cell))
    model.add(layers.Dense(output_dim, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[keras.metrics.BinaryAccuracy(name="binary_accuracy")])
    return model


def thresholds_for(y, probs):
    result = []
    for j in range(y.shape[1]):
        p, r, ts = precision_recall_curve(y[:, j], probs[:, j])
        if not len(ts): result.append(.5); continue
        f = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
        result.append(float(ts[int(np.argmax(f))]))
    return result


def score_predictions(y, probs, thresholds):
    pred = (probs >= np.asarray(thresholds)[None, :]).astype(int)
    return {"subset_accuracy": float(accuracy_score(y, pred)),
            "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "micro_precision": float(precision_score(y, pred, average="micro", zero_division=0)),
            "micro_recall": float(recall_score(y, pred, average="micro", zero_division=0))}, pred


def tune(name, xtr, ytr, xv, yv, matrices, output_dim):
    """RandomSearch compares LSTM/BiLSTM, units/dropout and embedding initialization."""
    candidates = []
    all_results = []
    for vectorization, matrix in matrices.items():
        tuner = kt.RandomSearch(lambda hp, m=matrix: build_model(hp, m, output_dim),
            objective="val_binary_accuracy", max_trials=TRIALS, executions_per_trial=1,
            directory=str(OUT / "keras_tuner"), project_name=f"{name}_{vectorization}",
            overwrite=True, seed=SEED)
        tuner.search(xtr, ytr, validation_data=(xv, yv), epochs=EPOCHS, batch_size=BATCH,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)], verbose=1)
        model = tuner.get_best_models(1)[0]
        hp = dict(tuner.get_best_hyperparameters(1)[0].values)
        trial_score = float(tuner.oracle.get_best_trials(1)[0].score)
        info = {"experiment": name, "vectorization": vectorization, "val_binary_accuracy": trial_score, "hyperparameters": hp}
        all_results.append(info); candidates.append((trial_score, model, vectorization, info))
    _, model, vectorization, info = max(candidates, key=lambda x: x[0])
    return {"model": model, "vectorization": vectorization, "result": info, "all_results": all_results}


def main():
    random.seed(SEED); np.random.seed(SEED); keras.utils.set_random_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = clean_frames(*load_data())
    
    strat = train[LABELS].sum(axis=1).clip(upper=3)
    try: tr, val = train_test_split(train, test_size=.1, random_state=SEED, stratify=strat)
    except ValueError: tr, val = train_test_split(train, test_size=.1, random_state=SEED)
    tr, val = tr.reset_index(drop=True), val.reset_index(drop=True)
    tokenizer, matrices = build_vectorizers(tr.processed_text.tolist())
    xtr, xv, xt = (encode(df.processed_text, tokenizer) for df in (tr, val, test))
    ytr, yv, yt = (df[LABELS].to_numpy(dtype="int32") for df in (tr, val, test))
    report = {"experiments": [], "test_metrics": {}}

    
    baseline = tune("baseline_all_labels", xtr, ytr, xv, yv, matrices, len(LABELS))
    base_thresholds = thresholds_for(yv, baseline["model"].predict(xv, verbose=0))
    base_metrics, _ = score_predictions(yt, baseline["model"].predict(xt, verbose=0), base_thresholds)
    report["experiments"] += baseline["all_results"]
    report["test_metrics"]["baseline_all_labels"] = base_metrics

    
    ytr_route = (ytr.sum(axis=1) > 1).astype("int32")
    yv_route = (yv.sum(axis=1) > 1).astype("int32")
    single_tr, multi_tr = ytr.sum(axis=1) == 1, ytr.sum(axis=1) > 1
    single_val, multi_val = yv.sum(axis=1) == 1, yv.sum(axis=1) > 1
    if min(single_tr.sum(), multi_tr.sum()) < 2:
        raise ValueError("Нужны минимум два обучающих примера и с одной, и с несколькими метками.")
    if not single_val.any() or not multi_val.any():
        raise ValueError("В validation должны быть примеры обеих групп; проверьте датасет/разбиение.")
    router = tune("router_single_vs_multi", xtr, ytr_route, xv, yv_route, matrices, 1)
    router_val_probs = router["model"].predict(xv, verbose=0).ravel()
    router_val_pred = (router_val_probs >= .5).astype(int)
    report["experiments"] += router["all_results"]
    report["test_metrics"]["router_validation"] = {
        "accuracy": float(accuracy_score(yv_route, router_val_pred)),
        "macro_f1": float(f1_score(yv_route, router_val_pred, average="macro", zero_division=0))}

    
    single = tune("specialist_single_label", xtr[single_tr], ytr[single_tr], xv[single_val], yv[single_val], matrices, len(LABELS))
    multi = tune("specialist_multi_label", xtr[multi_tr], ytr[multi_tr], xv[multi_val], yv[multi_val], matrices, len(LABELS))
    single_thresholds = thresholds_for(yv[single_val], single["model"].predict(xv[single_val], verbose=0))
    multi_thresholds = thresholds_for(yv[multi_val], multi["model"].predict(xv[multi_val], verbose=0))
    report["experiments"] += single["all_results"] + multi["all_results"]

    
    route_probs = router["model"].predict(xt, verbose=0).ravel()
    is_multi = route_probs >= .5
    routed = np.zeros_like(yt)
    idx_single, idx_multi = np.where(~is_multi)[0], np.where(is_multi)[0]
    if len(idx_single):
        probs = single["model"].predict(xt[idx_single], verbose=0)
        routed[idx_single, np.argmax(probs, axis=1)] = 1
    if len(idx_multi):
        probs = multi["model"].predict(xt[idx_multi], verbose=0)
        pred = (probs >= np.asarray(multi_thresholds)[None, :]).astype(int)
        empty = pred.sum(axis=1) == 0
        if empty.any(): pred[empty, np.argmax(probs[empty], axis=1)] = 1
        routed[idx_multi] = pred
    routed_metrics = {
        "subset_accuracy": float(accuracy_score(yt, routed)),
        "micro_f1": float(f1_score(yt, routed, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(yt, routed, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(yt, routed, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(yt, routed, average="micro", zero_division=0)),
        "router_accuracy": float(accuracy_score((yt.sum(axis=1) > 1).astype(int), is_multi.astype(int))) }
    report["test_metrics"]["routed_single_multi"] = routed_metrics
    best_approach = "routed_single_multi" if routed_metrics["micro_f1"] > base_metrics["micro_f1"] else "baseline_all_labels"
    report["best_approach"] = best_approach

    
    router["model"].save(OUT / "router_model.keras")
    single["model"].save(OUT / "single_label_model.keras")
    multi["model"].save(OUT / "multi_label_model.keras")
    if best_approach == "baseline_all_labels":
        baseline["model"].save(OUT / "model.keras")
        metadata = {"approach": best_approach, "vectorization": baseline["vectorization"], "thresholds": base_thresholds}
    else:
     
        metadata = {"approach": best_approach,
            "vectorization": {"router": router["vectorization"], "single_label": single["vectorization"], "multi_label": multi["vectorization"]},
            "route_threshold": .5, "single_thresholds": single_thresholds, "multi_thresholds": multi_thresholds,
            "components": {"router": "router_model.keras", "single_label": "single_label_model.keras", "multi_label": "multi_label_model.keras"}}
    metadata.update({"max_len": MAX_LEN, "labels": LABELS})
    (OUT / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "experiment_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(report["test_metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nСравнение завершено. Лучший подход:", best_approach)
    print(json.dumps(report["test_metrics"], ensure_ascii=False, indent=2))
    print("Артефакты сохранены в", OUT)

if __name__ == "__main__":
    main()
