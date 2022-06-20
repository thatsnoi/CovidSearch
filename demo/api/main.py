from typing import Union

from fastapi import FastAPI

app = FastAPI()

@app.get("/search")
def search(q: Union[str, None] = None):

