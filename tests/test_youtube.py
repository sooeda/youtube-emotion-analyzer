import pytest
from src.emotion_app.youtube import extract_video_id
@pytest.mark.parametrize("url,expected",[("https://www.youtube.com/watch?v=dQw4w9WgXcQ","dQw4w9WgXcQ"),("https://youtu.be/dQw4w9WgXcQ","dQw4w9WgXcQ"),("https://youtube.com/shorts/dQw4w9WgXcQ","dQw4w9WgXcQ")])
def test_extract_video_id(url,expected): assert extract_video_id(url)==expected
def test_reject_invalid_url():
    with pytest.raises(ValueError): extract_video_id("https://example.com/video")
