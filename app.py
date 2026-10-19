import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from src.emotion_app.youtube import extract_video_id, fetch_comments
from src.emotion_app.model import EmotionPredictor

ROOT=Path(__file__).resolve().parent
load_dotenv(ROOT/".env")
app=Flask(__name__)
_predictor=None

def predictor():
    global _predictor
    if _predictor is None: _predictor=EmotionPredictor(ROOT/"artifacts")
    return _predictor

@app.get("/")
def index(): return render_template("index.html")

@app.post("/analyze")
def analyze():
    try:
        payload=request.get_json(silent=True) or {}
        video_url=(payload.get("video_url") or "").strip()
        if not video_url: return jsonify({"error":"Укажите ссылку на YouTube-видео."}),400
        video_id=extract_video_id(video_url)
        comments=fetch_comments(video_id,int(os.getenv("MAX_COMMENTS","500")))
        if not comments: return jsonify({"error":"Комментарии не найдены или автор отключил комментарии."}),200
        results=predictor().predict(comments)
        counts={label:sum(label in item["emotions"] for item in results) for label in __import__("src.emotion_app.model",fromlist=["LABELS"]).LABELS}
        return jsonify({"video_id":video_id,"total_comments":len(results),"labels":list(counts),"values":list(counts.values()),"results":results[:100]})
    except Exception as exc:
        app.logger.exception("Analysis failed")
        return jsonify({"error":str(exc)}),500

if __name__=="__main__": app.run(debug=True)
