## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Named Entity Recognition is an important Natural Language Processing task used to identify entities such as person names, organizations, and locations from unstructured text. The objective is to develop an interactive prototype that accepts user text, identifies named entities using a trained NER model, and displays the detected entities through a user-friendly Gradio interface.
### DESIGN STEPS:

STEP 1:
Import the required Python libraries and load the environment variables containing the Hugging Face API key and API endpoint.

STEP 2:
Create a helper function to send the input text to the Hugging Face NER model API and receive the identified entities as output.

STEP 3:
Develop a Gradio-based user interface that accepts text input, processes it using the NER model, and highlights the identified entities. Merge subword tokens where necessary to display complete entity names.
### PROGRAM:
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

import requests, json
#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))

    API_URL = os.environ['HF_API_NER_BASE'] #NER endpoint
text = "My name is Andrew, I'm building DeepLearningAI and I live in California"
get_completion(text, parameters=None, ENDPOINT_URL= API_URL)

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    return {"text": input, "entities": output}
gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    #Here we introduce a new tag, examples, easy to use examples for your application
                    examples=["My name is rohith and I live in chennai", "My name is vijay and work at TCS"])
demo.launch(share=True, server_port=int(os.environ['PORT3']))
### OUTPUT:
<img width="1162" height="617" alt="Screenshot 2026-09-01 140339" src="https://github.com/user-attachments/assets/695dd698-18ee-4def-91d2-a91039f21be8" />

### RESULT:
Thus, the Named Entity Recognition prototype was successfully developed using the dslim/bert-base-NER model and deployed using the Gradio framework. The application successfully accepts user input and identifies and highlights named entities from the given text.
