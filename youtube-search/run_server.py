import subprocess
import time
from pyngrok import ngrok
import threading
import webbrowser

def run_fastapi():
    subprocess.run(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"])

# FastAPI 서버를 별도 스레드에서 실행
server_thread = threading.Thread(target=run_fastapi)
server_thread.daemon = True
server_thread.start()

print("FastAPI 서버가 시작되었습니다...")
time.sleep(3)  # 서버 시작 대기

# ngrok 터널 생성
public_url = ngrok.connect(8000).public_url
print(f"\n✨ ngrok 터널이 생성되었습니다!")
print(f"🌐 Public URL: {public_url}")
print(f"\n브라우저에서 위 URL로 접속하세요.")

try:
    # 서버 유지
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n서버를 종료합니다...")
    ngrok.kill()