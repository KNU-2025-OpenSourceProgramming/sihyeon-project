import os

# 디렉토리 구조 생성
directories = [
    'www',
    'www/static',
    'www/static/js',
    'www/static/css',
    'www/templates'
]

for dir in directories:
    os.makedirs(dir, exist_ok=True)
    print(f"Created: {dir}")