import os
import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse

from GigaChat.GigaChat_RAG import set_environ, get_answer


app = FastAPI()


@app.get("/")
async def main():
    return FileResponse("public/answer.html")


@app.post("/api/answer")
def get_answers(data=Body()):
    answer = get_answer(data["question"])
    return answer["result"]


if __name__ == "__main__":
    set_environ()
    uvicorn.run(app, host="0.0.0.0", port=8000)
