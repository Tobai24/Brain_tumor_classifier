from fastapi import FASTAPI

app = FASTAPI()

@app.get("/")
def greet():
    print("hello world")