# gov-data-buddy

The project uses form recognizer to parse the input file,
followed by using Langchain for doc Q&A. The app itself is built
with Streamlit. <br>

To run the code, you need to provide a .env file with the following variables:
```
OPENAI_DEPLOYMENT_ENDPOINT
OPENAI_API_KEY
OPENAI_DEPLOYMENT_NAME
OPENAI_DEPLOYMENT_VERSION
OPENAI_MODEL_NAME

OPENAI_EMBEDDING_DEPLOYMENT_NAME
OPENAI_EMBEDDING_MODEL_NAME

FORM_RECOGNIZER_ENDPOINT
FORM_RECOGNIZER_KEY
```
Next, run `python indexer.py`, which will index the documents in the `data/documentation` folder, 
to a new folder, `data/dbs`. <br>

Finally, run `streamlit run app.py` to run the app. <br>

Notes: you may need to provide the full path of the `dbs/documentation/faiss` directory created
by `indexer.py` to the app in `app.py`.# gov-data-buddy
