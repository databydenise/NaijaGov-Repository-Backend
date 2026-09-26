from fastapi import FastAPI, HTTPException

from src.exceptions.handlers import general_exception_handler, http_exception_handler
from src.middlewares.response import response_transformer

# Initialise application server
app = FastAPI()


app.add_exception_handler(
    HTTPException,
    http_exception_handler,  # type: ignore  # noqa: PGH003
)

app.add_exception_handler(
    Exception,
    general_exception_handler,
)

# Transform all responses to a standard format
app.middleware("http")(response_transformer)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "Okay"}
