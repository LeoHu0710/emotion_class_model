import os
import torch
import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from log import api_logger

mapping_table = {0 : "sadness", 1 : "joy", 2 : "love", 3 : "anger", 4 : "fear", 5 : "surprise"}

checkpoint_path = os.path.join("results","checkpoint_results")

## loading model and tokenizer
model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

app = FastAPI(
    title = "Sentiment Analysis api",
    description = "輸入英文句子並輸出情緒類別"
)


@app.get("/root")
def root():
    """
    測試 api 是否有通
    """
    api_logger.info("get root test !!")
    return {"message": "get root test !!"}


class SentenceRequest(BaseModel):
    sentence: str

@app.post("/sentiments")
def sentiments(req: SentenceRequest):
    """
    獲取 英文句子
     
    - **sentence** : str
    
    返回 輸入的英文句子和預測情緒類別
    """
    try :
        sentence = req.sentence

        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits

        predicted_class = torch.argmax(logits, dim=1).item()
        predict = mapping_table[predicted_class]
        response = {'sentence': sentence, 'Predicted_category': predict}
        api_logger.info(response)
        return JSONResponse(status_code = 200, content = response)
    
    except Exception as e :
        return JSONResponse(status_code = 500, content = f"Error : {e}")

async def main():
    ## Setting uvicorn config
    config = uvicorn.Config("main:app", host = "0.0.0.0", port = 8080, reload = True)
    server = uvicorn.Server(config)
    await server.serve()

## start Uvicorn server
if __name__ == "__main__":
    api_logger.info(f"Starting Sentiment Analysis api. host: 0.0.0.0, port: 8080")
    asyncio.run(main())