from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import uvicorn
import os

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="www/static"), name="static")
templates = Jinja2Templates(directory="www/templates")

# YouTube API 키 (여기에 실제 API 키 입력)
YOUTUBE_API_KEY = "AIzaSyAc5VpXXFeVBQx8BbioZQ_I1TTNpao7-0M"
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/search/{query}")
async def search_youtube(query: str, max_results: int = 10):
    try:
        search_response = youtube.search().list(
            q=query,
            part='snippet',
            maxResults=max_results,
            type='video'
        ).execute()
        
        videos = []
        for item in search_response.get('items', []):
            video_data = {
                'id': item['id']['videoId'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'thumbnail': item['snippet']['thumbnails']['high']['url'],
                'channel': item['snippet']['channelTitle'],
                'publishedAt': item['snippet']['publishedAt']
            }
            videos.append(video_data)
            
        return JSONResponse(content={"results": videos})
    except HttpError as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/video/{video_id}")
async def get_video_details(video_id: str):
    try:
        video_response = youtube.videos().list(
            id=video_id,
            part='snippet,statistics,contentDetails'
        ).execute()
        
        if video_response['items']:
            video = video_response['items'][0]
            video_data = {
                'id': video['id'],
                'title': video['snippet']['title'],
                'description': video['snippet']['description'],
                'thumbnail': video['snippet']['thumbnails']['high']['url'],
                'channel': video['snippet']['channelTitle'],
                'publishedAt': video['snippet']['publishedAt'],
                'viewCount': video['statistics'].get('viewCount', '0'),
                'likeCount': video['statistics'].get('likeCount', '0'),
                'commentCount': video['statistics'].get('commentCount', '0'),
                'duration': video['contentDetails']['duration']
            }
            return JSONResponse(content=video_data)
        else:
            return JSONResponse(content={"error": "Video not found"}, status_code=404)
    except HttpError as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)