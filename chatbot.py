import csv
import datetime
import re
from datetime import datetime
import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments


conversations = []
converted_data = []
def preprocess_chat_data(file_path):
    
    current_conversation = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                # Extract the timestamp, sender, and message
                timestamp_end = line.find(" - ")
                if timestamp_end != -1:
                    timestamp = line[:timestamp_end]
                    sender_message = line[timestamp_end + 3:]
                    
                    # Extract day, date, time, am/pm from the timestamp
                    timestamp_parts = timestamp.split(", ")
                    day_date = timestamp_parts[0]
                    time = timestamp_parts[1].strip()
                    am_pm = time[-2:].lower()
                    time = time[:-2].strip()
                    
                    # Parse the day, date, time into a datetime object
                    timestamp_obj = datetime.strptime(day_date + " " + time + " " + am_pm, "%d/%m/%y %I:%M %p")
                    
                    # Extract the sender and message
                    sender_end = sender_message.find(": ")
                    if sender_end != -1:
                        sender = sender_message[:sender_end]
                        message = sender_message[sender_end + 2:]
                        current_conversation.append((timestamp_obj, sender, message))
                elif current_conversation:
                    conversations.append(current_conversation)
                    current_conversation = []

        if current_conversation:
            conversations.append(current_conversation)

    except Exception as e:
        print("An error occurred during chat data processing:", e)

    return conversations


# conversations = preprocess_chat_data("chat_stat_Harsh.txt")

def convert_data(conversations):
    for item in conversations:
        converted_item = [str(elem) for elem in item]
        converted_data.append(converted_item)
    return converted_data

# conversations = convert_data(conversations)

# Define the conversation dataset class
class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        encoded_inputs = self.tokenizer.encode_plus(
            conversation,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )
        
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

# Example conversation list
# conversations = [
#     "11/05/23, 7:40 pm - Utkarsh Kumar Sahu: Poore do ghante lag gaye banane me",
#     "12/05/23, 8:01 am - Harsh Cer: Noobs",
#     "12/05/23, 8:02 am - Utkarsh Kumar Sahu: Haha",
#      .....
# ]

# Load the pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add a padding token to the tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Create the conversation dataset
dataset = ConversationDataset(converted_data, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./chat_model',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
)

# Define a custom Trainer class
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        return loss

# Create a CustomTrainer instance
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=lambda data: {'input_ids': torch.stack([item['input_ids'] for item in data]),
                               'attention_mask': torch.stack([item['attention_mask'] for item in data])}
)

# Train the model
trainer.train()

# Save the model
trainer.save_model('./chat_model')
tokenizer.save_pretrained("./chat_model")


# Load the trained model and tokenizer
model_path = "./chat_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)


# Set the device to use (e.g., "cuda" for GPU or "cpu" for CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Disable the pad_token_id and attention_mask warning
model.config.pad_token_id = model.config.eos_token_id
model.config.use_cache = False

# Function to generate chat responses
def generate_response(input_text, max_length=10):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate text based on the input
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

