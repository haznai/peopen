# %% server imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import RootModel
from uvicorn import run

import requests

# %% model imports
from model_definition import (
    PenPrompterNetwork,
    Dataset,
    LanguageModel,
    get_model_path,
    get_train_and_valid_path,
)

from submodels.factual_consistency_model import (
    FactualConsistencyNetwork,
)


# %% misc imports
import os
import random

# %% model setup
model_path_pp = get_model_path("2024-10-26_penprompter-first-full-training.json")
model_path_fc = get_model_path("2024-10-25_factual_consistency.json")

fc_model = FactualConsistencyNetwork()
fc_model.load(model_path_fc)
fc_model._compiled = True
network = PenPrompterNetwork(fc_model)
trained_network = PenPrompterNetwork(fc_model)
trained_network.load(model_path_pp)
lm = LanguageModel(
    name="gpt-4o-mini-2024-07-18",
    url="https://api.openai.com/v1/",
    api_key=os.environ.get("OPENAI_API_KEY"),  # type: ignore
)

train_path, valid_path = get_train_and_valid_path()
data = Dataset(train_pickle_path=str(train_path), valid_pickle_path=str(valid_path))
trainset = data.data["train"]
valset = data.data["validation"]


# %% server setup
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# %% endpoint definitions
class AceEditEvent(RootModel):
    root: dict


@app.post("/aceEditEvent")
async def handle_ace_edit_event(event: AceEditEvent):
    text_value = ""
    content = event.root["content"]
    if content and isinstance(content[0], dict):
        text_value = content[0].get("value", "")
        print(f"Received text: {text_value}")
    else:
        print("No valid content received")
    if text_value == "":
        return {"message": "No valid content received"}
    return trained_network.get_final_prediction(text_value)


@app.options("/aceEditEvent")
async def options_ace_edit_event():
    return {"message": "OK"}


@app.options("/documentReady")
async def options_document_ready():
    return {"message": "OK"}


@app.get("/documentReady")
async def handle_document_ready_event():
    # get random example from train/validation set
    # @todo: make this actually pickable from `ep_peopens` left box

    # random example from validation set first
    random_index = random.randint(0, len(valset) - 1)
    random_example = valset[random_index]

    titel = random_example["titel"]
    im_detail = random_example["im_detail"]
    argumenteKomitee = random_example["argumenteKomitee"]
    empfehlungKomitee = random_example["empfehlungKomitee"]
    argumenteBundesrat = random_example["argumenteBundesrat"]
    empfehlungBundesrat = random_example["empfehlungBundesrat"]

    draft_text = trained_network.get_first_draft(
        titel=titel,
        im_detail=im_detail,
        argumenteKomitee=argumenteKomitee,
        empfehlungKomitee=empfehlungKomitee,
        argumenteBundesrat=argumenteBundesrat,
        empfehlungBundesrat=empfehlungBundesrat,
    )

    try:
        # @todo: get padID, don't hardcode 'test'
        # @todo: get api_key, don't hardcode 'ayy'
        _ = requests.post(
            "http://localhost:9001/api/1/setText",
            params={"api_key": "ayy", "padID": "test"},
            data={"text": draft_text},
        )
    except Exception as e:
        print(f"Failed to set pad text: {e}")

    return draft_text


# %% run server
if __name__ == "__main__":
    run(app, host="localhost", port=50627)
