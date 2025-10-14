from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.presentation.v1.controllers.root_router import root_router
import fastapi_problem.handler


app = FastAPI()
eh = fastapi_problem.handler.new_exception_handler()
fastapi_problem.handler.add_exception_handler(app, eh)

app.include_router(root_router)


