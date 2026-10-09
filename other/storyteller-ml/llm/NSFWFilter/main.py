
from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
import asyncio
app = FastAPI()
import json

class LLMResponse(BaseModel):
    result:int
    
class LLMRequest(BaseModel):
    text:str
    
llm = Llama(
    model_path="mistral-ft-optimized-1227.Q5_K_M.gguf",
    chat_format="mistral-instruct",
    n_gpu_layers=100, # Uncomment to use GPU acceleration
    seed=1337, # Uncomment to set a specific seed
    n_ctx=2048, # Uncomment to increase the context window
)

tasks = asyncio.Queue()

@app.post("/tasks/")
async def create_task_classify_NSFW(background_tasks: BackgroundTasks):
    task_id = "some_unique_id"
    await tasks.put(task_id)
    background_tasks.add_task(process_task, task_id)
    return {"message": "Task received", "task_id": task_id}

async def process_task(task_id):
    pass
    
@app.post("/classify-NSFW", response_model=LLMResponse)
async def classify_NSFW(request: LLMRequest) -> LLMResponse:
    # For demonstration, count the number of characters in the input text.
    result = len(request.text)
    return LLMResponse(result=result)
    
def classify_text(text:str,debug:bool = False):
    nsfw_filter_prompt = """<s>[Inst] You are a super brilliant well read assistant, given a string of text as input classify the text as not safe for work or safe for work. The class scale should follow the 4 levels of classification. 1. Safe for work 2. Passable for work 3. Sort of not safe for work 4. Definitely not safe for work. Return this as json object it should look like { "class": <class scale> }. Do not explain your answer just the json output is enough. Input: {{ }}.[\Inst]"""
    nsfw_filter_prompt = nsfw_filter_prompt.replace("{{ }}",text)
    
    output = llm(
      grammar=LlamaGrammar.from_file("json.gbnf"),
      prompt=nsfw_filter_prompt, # Prompt completion, can also call create_completion
      max_tokens=None, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["\n","</s>"], # Stop generating just before the model would generate a new question
      echo=False # Echo the prompt back in the output
    ) 
    res = ""
    if debug == True:
        for response in output:
            res += response["choices"][0]["text"]
    else:
        res += output["choices"][0]["text"]   
    json_object = json.loads(res)
    return json_object

def test_large_file():
    s = 0
    with open('output.txt',mode="w") as w:
        with open('output.sql') as f:
            lines = f.readlines() 
            for idx,line in enumerate(lines):
                if s < 100:
                    try:
                        print(f"{idx}:{line}")
                        res = classify_text(line)
                        value = res["class"]
                        w.writelines(f"{idx}:{value}|{line}")
                        s += 1
                    except Exception:
                        pass
                    
def main():
   test_large_file()
            
if __name__ == "__main__":
    main()
    
# uvicorn main:app --reload
# curl -X 'POST' \
#   'http://localhost:8000/classify-NSFW' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{"text":"Hello, World!"}'