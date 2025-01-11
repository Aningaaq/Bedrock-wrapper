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
  response = bedrock.invoke_model(
      modelId='amazon.titan-tg1-large',
      contentType='text/plain',
      accept='text/plain',
      # body is the prompt we send
      body=prompt,
      # Additional parameters to guide the generation
      # (e.g. limit token output, zero randomness, etc.)
      maxTokens=2,        # Keep it small to limit the output
      temperature=0,      # Zero randomness
      topP=1,             # Use standard decoding
      stopSequences=["\n"]  # Stop at new line
  )

  # The response body is a streaming output (a file-like object in boto3)
  model_output = response['body'].read().decode('utf-8')

  # Strip extra whitespace (and any unexpected text)
  model_output = model_output.strip()

  # In practice, you might want to post-process or validate
  # to ensure you only return "true" or "false".
  return model_output

@app.route("/is-question", method=["GET"])
def is_question():
  text = request.args.get("text", default="")

  is_question = classify_question_bedrock(text)
  
  return is_question


if __name__ == "__main__":
  # It's generally recommended to run via poetry run python -m flask run
  # But for simplicity, we'll just do app.run for this example.
  app.run(host="0.0.0.0", port=5000, debug=True)