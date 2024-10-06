# %% run server
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import RootModel
from uvicorn import run

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class AceEditEvent(RootModel):
    root: dict


@app.post("/aceEditEvent")
async def handle_ace_edit_event(event: AceEditEvent):
    content = event.root["content"]
    if content and isinstance(content[0], dict):
        text_value = content[0].get("value", "")
        print(f"Received text: {text_value}")
    else:
        print("No valid content received")
    print(text_value)

    return network.get_final_prediction(text_value)


@app.options("/aceEditEvent")
async def options_ace_edit_event():
    return {"message": "OK"}


@app.options("/postAceInit")
async def options_post_ace_init():
    return {"message": "OK"}


@app.get("/postAceInit")
async def handle_post_ace_init_event():
    return network.get_draft_wortlaut_prediction(
        titel,
        im_detail,
        argumenteKomitee,
        empfehlungKomitee,
        argumenteBundesrat,
        empfehlungBundesrat,
    )


if __name__ == "__main__":
    run(app, host="localhost", port=50627)
