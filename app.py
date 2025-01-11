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
  embedding_vector = response["embedding"]
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


if __name__ == "__main__":
  # It's generally recommended to run via poetry run python -m flask run
  # But for simplicity, we'll just do app.run for this example.
  app.run(host="0.0.0.0", port=5000, debug=True)