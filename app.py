from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.responses import RedirectResponse
from text_summarizer.pipeline.prediction import PredictionPipeline

app = FastAPI(title="Text Summarizer")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "summary": ""}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    obj = PredictionPipeline()
    summary = obj.predict(text)
    
    # Clean up the summary text
    # Replace <n> with actual newlines, and clean up any other formatting issues
    summary = summary.replace("<n>", "\n").replace("</n>", "")
    # Replace multiple spaces with single space (except after periods)
    summary = re.sub(r' +', ' ', summary)
    # Remove multiple consecutive newlines (keep max 2 for paragraph breaks)
    summary = re.sub(r'\n{3,}', '\n\n', summary)
    # Clean up any trailing/leading whitespace on each line
    lines = [line.strip() for line in summary.split('\n')]
    summary = '\n'.join(lines)
    # Strip leading/trailing whitespace from entire summary
    summary = summary.strip()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "summary": summary,
            "original": text
        }
    )

@app.get("/train")
async def training():
    os.system("python main.py")
    return {"status": "Training completed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
