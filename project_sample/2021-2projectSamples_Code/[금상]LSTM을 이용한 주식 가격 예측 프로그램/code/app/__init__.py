# file name : __init__.py
# pwd : /project_name/app/__init__.py
 
from flask import Flask
# 추가할 모듈이 있다면 추가
 
app= Flask(__name__)
 
# 파일 이름이 index.py이므로
from app.main.index import main as main
 
# 위에서 추가한 파일을 연동해주는 역할
app.register_blueprint(main)# as main으로 설정해주었으므로
