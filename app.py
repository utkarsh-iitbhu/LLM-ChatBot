from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from chatbot import generate_response
app = Flask(__name__)


# Load the trained model and tokenizer
model_path = "./chat_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

@app.route("/")
def home():
    return render_template("index.html")  # Create an HTML template for the home page

@app.route("/process_message", methods=["POST"])
def process_message():
    try:
        # user_input = request.form["user_input"]
        # Tokenize the user input
        user_input = request.form["user_input"]
        response = generate_response(user_input)
        return response
        # input_ids = tokenizer.encode(user_input, return_tensors="pt")

        # # Generate a response
        # with torch.no_grad():
        #     output = model.generate(input_ids, max_length=10, num_return_sequences=1)
        #     response = generate_response(user_input)

        # response = tokenizer.decode(output[0], skip_special_tokens=True)

        # return response

    except Exception as e:
        print("An error occurred while processing the message.")
        print(str(e))
        return ""


if __name__ == "__main__":
    app.run(debug=True)
