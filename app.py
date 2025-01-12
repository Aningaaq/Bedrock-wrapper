from flask import Flask, request, jsonify
import boto3
import json
from botocore.config import Config

app = Flask(__name__)

@app.route("/")
def hello_world():
  return "Hello, Flask with Poetry!"


bedrock = boto3.client(service_name='bedrock-runtime', config=Config(
  region_name = 'us-east-1',
))

def get_embeddings_bedrock(text):
  accept = "application/json"
  content_type = "application/json"
  body = json.dumps({ "inputText": text }).encode('utf-8')

  response = bedrock.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=body,
    accept=accept, contentType=content_type
  )

  # response["body"] is a StreamingBody object
  response_body = response["body"].read()         # Read the stream (bytes)
  response_str = response_body.decode("utf-8")    # Decode to a string
  response_json = json.loads(response_str)        # Parse string as JSON

  print("response json : ", response_json)
  # Now you can access the JSON keys
  embedding_vector = response_json["embeddingsByType"]["float"]
  print("embed vector: ", embedding_vector)
  return embedding_vector

# New route to get embeddings
@app.route("/question-embedding", methods=["GET"])
def question_embedding():
  # 1. Retrieve 'text' from the query parameter.
  #    Example request: GET /question-embedding?text=Hello
  text = request.args.get("text", default="")

  print("text : " + text)
  # 2. Call the helper function to get embeddings.
  embedding_vector = get_embeddings_bedrock(text)


  print("emebdding_vector : " + str(embedding_vector))
  # 3. Return the embedding vector as JSON.
  return jsonify({"embedding": embedding_vector})

PROMPT_TEMPLATE = """You are a text classification model.
Classify the following text as a question or not.
If the text is a question, output 'true'. Otherwise, output 'false'.

Text: "{text}"
Answer:
"""

def classify_question_bedrock(text):
  prompt = PROMPT_TEMPLATE.format(text=text)
  # Invoke the Amazon Titan model (change modelId if desired)
  body = json.dumps({
      "inputText": prompt,
      "textGenerationConfig": {
        "maxTokenCount": 2,         # severely limit the length
        "temperature": 0,          # reduce randomness
        "topP": 1,                 # typical decoding
      }
  })
  
  response = bedrock.invoke_model(
      modelId='amazon.titan-tg1-large',
      contentType='application/json',
      accept='application/json',
      body=body
  )

  # The response is a streaming body that we read, then parse as JSON
  raw_output = response['body'].read().decode('utf-8')
  output_json = json.loads(raw_output)

  # Titan responses typically look like:
  # {
  #   "results": [
  #       {"outputText": "true"}
  #   ]
  # }
  generated_text = output_json["results"][0]["outputText"].strip()

  # In practice, you might want to ensure the output is strictly 'true' or 'false'
  # or handle any unexpected text.
  return generated_text

@app.route("/is-question", methods=["GET"])
def is_question():
  text = request.args.get("text", default="")

  is_question = True  if classify_question_bedrock(text) == "True" else False
  response  = {
    "is_question": is_question
  }
  return json.dumps(response)


if __name__ == "__main__":
  # It's generally recommended to run via poetry run python -m flask run
  # But for simplicity, we'll just do app.run for this example.
  app.run(host="0.0.0.0", port=5000, debug=True)