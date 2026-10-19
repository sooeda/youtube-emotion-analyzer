import json, os, urllib.error, urllib.parse, urllib.request

API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

def extract_video_id(url: str) -> str:
    from urllib.parse import urlparse, parse_qs
    import re
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch": video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            match = re.match(r"/(?:embed|shorts|live)/([^/?]+)", parsed.path)
            video_id = match.group(1) if match else ""
    else: video_id = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id or ""):
        raise ValueError("Не удалось определить ID видео. Проверьте ссылку YouTube.")
    return video_id

def fetch_comments(video_id: str, max_comments: int = 500) -> list[str]:
    key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not key or key == "put_your_key_here": raise RuntimeError("Укажите YOUTUBE_API_KEY в файле .env.")
    comments, page_token = [], None
    while len(comments) < max_comments:
        params = {"part":"snippet", "videoId":video_id, "maxResults":min(100,max_comments-len(comments)), "textFormat":"plainText", "key":key}
        if page_token: params["pageToken"] = page_token
        req = urllib.request.Request(API_URL + "?" + urllib.parse.urlencode(params), headers={"User-Agent":"YouTubeEmotionAnalysis/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as response: payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try: message = json.loads(detail).get("error", {}).get("message", detail)
            except json.JSONDecodeError: message = detail
            raise RuntimeError(f"YouTube API HTTP {exc.code}: {message}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Не удалось подключиться к YouTube API: {exc}") from exc
        for item in payload.get("items", []):
            snip = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            text = snip.get("textDisplay", "").strip()
            if text: comments.append(text)
        page_token = payload.get("nextPageToken")
        if not page_token: break
    return comments
