from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, World!"}

@app.get("/api/post")
def get_posts():
    return {"posts": ["Post 1", "Post 2", "Post 3"]}