import uvicorn
from fastapi import FastAPI
import joblib

gender_vectorizer = open("models/gender_vectorizer.pkl", "rb")
gender_cv = joblib.load(gender_vectorizer)

gender_nv_model =  open("models/gender_nv_model.pkl", "rb")
gender_clf = joblib.load(gender_nv_model)

app = FastAPI()

@app.get('/')
async def index():
    return { 'messamige' : 'ML전문가님 안녕하세요.'}

@app.get('/items/{name}')
async def get_items(name):
    return { 'name' : name }

@app.get('/predeict')
async def predict(name):
    vactorized_name = gender_cv.transform([name]).toarray()
    prediction = gender_clf.predict(vactorized_name)

    if prediction[0] == 0:
        result = "여성"
    else :
        result = "남성"

    return { 'origin name' : name, "예측" : result }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    # uvicorn main:app --reload로 실행
    # 웹에 http://127.0.0.1:8000/docs로 검색

    