#!/usr/bin/env python3

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, numpy as np
from torch.quantization import quantize_dynamic
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
import re, pandas as pd
from sklearn.model_selection import train_test_split
import gc, json, os


def preprocess_text(text):
    """
    Basic text cleaning function.
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    text = re.sub(r"\@\w+|\#", "", text)
    return text

def preprocess_dataframe(df, columns):
    """
    Preprocess the dataframe by applying text cleaning.
    Args:
    - df (pd.DataFrame): DataFrame with text data.
    - columns (list): List of columns to preprocess.
    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """
    # Remove duplicates
    df = df.drop_duplicates()

    # Drop rows with any NaN values
    df = df.dropna()

    # Clean the text
    for col in columns:
        df[col] = df[col].apply(preprocess_text)

    return df

def load_dataset(file_path, rows):
    df = pd.read_csv(file_path)
    print(f"Dataset loaded. Number of rows: {len(df)}")
    df = df[["question", "answer"]]

    # Shuffling the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    # Limiting to the first 1000 rows
    df = df.head(rows)

    df = preprocess_dataframe(df, ["question", "answer"])

    return df

def format_qa(conv):
    messages = [
        {"role": "user", "content": conv["question"]},
        {"role": "assistant", "content": conv["answer"]},
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False).strip()
    return {"text": chat}

def split_dataset(df):
    # Split the DataFrame first
    # Assuming df is your DataFrame containing the dataset
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)  # Adjust the test_size as needed
    train_df, val_df = train_test_split(train_val_df, test_size=0.18, random_state=42)  # Adjust to get ~10-15% validation data

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Testing set size: {len(test_df)}")

    train_dataset = train_df.apply(format_qa, axis=1)
    val_dataset = val_df.apply(format_qa, axis=1)
    test_dataset = test_df.apply(format_qa, axis=1)

    return train_dataset, val_dataset, test_dataset

def dynamic_quantization(model):
    quantization_start = time.time()
    model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).to(device)
    quantization_end = time.time()
    print(f"Model quantized in {quantization_end - quantization_start:.2f} seconds")

    return model

def extract_substring(input_string):
    user_index = input_string.find("user:\n")
    if user_index == -1:
        return None  # "user:\n" not found
    assistant_index = input_string.find("assistant:", user_index)
    if assistant_index == -1:
        return None  # "assistant:" not found after "user:\n"
    question = input_string[user_index + len("user:\n"):assistant_index].strip()
    answer = input_string[assistant_index + len("assistant:\n"):].strip()
    return question, answer

def generate_response(question, model, tokenizer):
    """
    Generate a response for a given question using the fine-tuned model.
    """
    # Prepare the question for the model
    print("Question: ", question)
    inputs = tokenizer.encode(question, return_tensors='pt').to(device)
    print("Question Encoded...")

    # Generate a response
    generation_start = time.time()
    output = model.generate(inputs, no_repeat_ngram_size=3, early_stopping=True, num_beams=5)
    generation_end = time.time()
    print(f"Response generated in {generation_end - generation_start:.2f} seconds")

    # Decode and print the response
    decoding_start = time.time()
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    decoding_end = time.time()
    print(f"Answer decoded in {decoding_end - decoding_start:.2f} seconds")

    return response

def generate_and_save_questions(n, test_dataset, model, tokenizer, model_name):
    # Shuffle the dataset
    # test_dataset = test_dataset.shuffle()

    # Create a filename based on the current timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"Saved Responses/questions_{timestamp}.json"

    # Initialize a list to hold our question-answer-response dictionaries
    questions_responses = []
    
    
    # Ensure directory exists, create if not
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    successful_responses = 0

    for i in range(n):
        # Get a question from the shuffled dataset
        data = test_dataset.iloc[i]  # use .iloc to access the Series
        print(data)
        question, answer = extract_substring(data['text'])
        if question:
            print("Extracted substring:", question, "\n")
            print("Answer:", answer)
        else:
            print("Substring not found")
            continue

        # Generate response
        response_start = time.time()
        try:
            with torch.no_grad():
                response = generate_response(question, model, tokenizer)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA OOM occurred!")
                print("OOM Question: ", question)
                print("OOM Question Length: ", len(question))

                questions_responses.append({
                    "question": question,
                    "answer": answer,
                    "generated_response": "CUDA OOM",
                    "response_time": "",
                    "question_number": i,
                    "successful": False
                })

                # Clear the CUDA cache
                torch.cuda.empty_cache()  
                gc.collect()
                continue
            else:
                # If the error is not a CUDA OOM, re-raise the exception
                raise e
        response_end = time.time()
        print(f"Response {i}/{n} generated.")
        successful_responses += 1

        # Append this question's details to our list
        questions_responses.append({
            "question": question,
            "answer": answer,
            "generated_response": response,
            "response_time": f"{response_end - response_start:.2f} seconds",
            "question_number": i,
            "successful": True
        })
        
        # Save our list of dictionaries to a JSON file
        with open(filename, "w") as file:
            json.dump({
                "metadata": {
                    "model_name": model_name,
                    "timestamp": timestamp,
                    "total_questions": n,
                    "successful_responses": successful_responses
                },
                "data": questions_responses
            }, file, indent=4)

        print(f"Data saved to {filename}")

        # # Write question, answer, and response time to the file
        # file.write(f"Question {i+1}:\n")
        # file.write(f"Question: {question}\n")
        # file.write(f"Answer: {answer}\n")
        # file.write(f"Generated Response: {response}\n")

        # file.write(f"Response Time: {response_end - response_start:.2f} seconds\n\n")

        del response  
        torch.cuda.empty_cache()  
        gc.collect()

    print(f"{n} questions generated and saved to {filename}")

    end_time = time.time()

    print(f"Total Time to respond to {n} questions was {(end_time-start_time)//60} minutes and {(end_time-start_time)%60:.2f} seconds")

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def calculate_perplexity(model, tokenizer, text):
    tokenize_input = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    loss = model(tokenize_input.input_ids, labels=labels.input_ids).loss
    return torch.exp(loss).item()


def calculate_and_compare_perplexity(model, tokenizer, train_df, test_df):
    print("Starting Perplexity Test")
    results_dir = os.path.join("results", "other")
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, "perplexity.txt")
    with open(results_path, 'w') as results_file:
        train_avg_perplexity = perplexity_df(model, tokenizer, train_df, results_file, 'Train')
        test_avg_perplexity = perplexity_df(model, tokenizer, test_df, results_file, 'Test')
        
        results_file.write(f"Average Perplexity for Training Data: {train_avg_perplexity}\n")
        results_file.write(f"Average Perplexity for Testing Data: {test_avg_perplexity}\n")
        results_file.write(f"Difference in Average Perplexity: {abs(train_avg_perplexity - test_avg_perplexity)}\n")

        print(f"Average Perplexity for Training Data: {train_avg_perplexity}")
        print(f"Average Perplexity for Testing Data: {test_avg_perplexity}")
        print(f"Difference in Average Perplexity: {abs(train_avg_perplexity - test_avg_perplexity)}")
    
    print("Ending Perplexity Test")

def perplexity_df(model, tokenizer, df, results_file, dataset_name):
    total_perplexity = 0
    for index, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]
        
        answer_perplexity = calculate_perplexity(model, tokenizer, answer)
        total_perplexity += answer_perplexity
        
        results_file.write(f"{dataset_name} Question: {question}\n")
        results_file.write(f"{dataset_name} Answer Perplexity: {answer_perplexity}\n\n")
        
    avg_perplexity = total_perplexity / len(df)
    return avg_perplexity

def calculate_bleu(reference, candidate):
    """
    Calculate the BLEU score for a single reference and a candidate sentence.
    """
    reference_tokens = word_tokenize(reference.lower())
    candidate_tokens = word_tokenize(candidate.lower())
    score = sentence_bleu([reference_tokens], candidate_tokens)
    return score

def test_bleu_score(json_file_path):
    """
    Test the BLEU score for responses in a JSON file.
    """
    # Load your JSON data
    print("Starting BLEU Test")
    data = load_json_data(json_file_path)

    # Ensure the results directory exists
    timestamp = data['metadata']['timestamp']
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare to save the results
    results_path = os.path.join(results_dir, "bleu_scores.json")
    print("BLEU Score Test Start:")
    bleu_total = 0
    bleu_results = []
    num_tests = 0

    # Iterate over your data to compute BLEU scores for each response
    for item in data["data"]:
        if item["successful"]:
            answer = item["answer"]
            generated_response = item["generated_response"]
            
            bleu_score = calculate_bleu(answer, generated_response)
            bleu_total += bleu_score  # Append blue score to the list
            num_tests += 1
            
            # Update metadata with average BLEU score
            metadata = data['metadata']
            metadata['average_bleu_score'] = bleu_total / num_tests
            
            # Store relevant information in a dictionary
            bleu_result = {
                "question_number": item["question_number"],
                "question": item["question"],
                "answer": answer,
                "generated_response": generated_response,
                "bleu_score": bleu_score
            }

            bleu_results.append(bleu_result)
            
            # Write the results to a JSON file
            with open(results_path, 'w') as results_file:
                json.dump({"metadata": metadata, "data": bleu_results}, results_file, indent=4)
    
    print("BLEU Test Ended")


def calculate_rouge(reference, candidate):
    """
    Calculate the ROUGE scores (ROUGE-1, ROUGE-2, and ROUGE-L) for a single reference and a candidate sentence.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

def test_rouge_score(json_file_path):
    """
    Test the ROUGE score for responses in a JSON file.
    """
    print("Starting ROUGE Test")
    # Load your JSON data
    data = load_json_data(json_file_path)

    # Ensure the results directory exists
    timestamp = data['metadata']['timestamp']
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare to save the results
    results_path = os.path.join(results_dir, "rouge.json")
    rouge_results = []
    rouge_1_total = 0
    rouge_2_total = 0
    rouge_l_total = 0
    num_tests = 0

    # Iterate over your data to compute ROUGE scores for each response
    # Iterate over your data to compute ROUGE scores for each response
    for item in data["data"]:
        if item["successful"]:
            answer = item["answer"]
            generated_response = item["generated_response"]

            
            rouge_scores = calculate_rouge(answer, generated_response)
            
            # Update running totals for averages
            rouge_1_total += rouge_scores['rouge1'].fmeasure
            rouge_2_total += rouge_scores['rouge2'].fmeasure
            rouge_l_total += rouge_scores['rougeL'].fmeasure
            num_tests += 1
            
            # Update metadata with averages
            metadata = data['metadata']
            metadata['average_rouge_1'] = rouge_1_total / num_tests
            metadata['average_rouge_2'] = rouge_2_total / num_tests
            metadata['average_rouge_l'] = rouge_l_total / num_tests
            
            # Store relevant information in a dictionary
            rouge_result = {
                "question_number": item["question_number"],
                "question": item["question"],
                "answer": answer,
                "generated_response": generated_response,
                "rouge_1": rouge_scores['rouge1'].fmeasure,
                "rouge_2": rouge_scores['rouge2'].fmeasure,
                "rouge_l": rouge_scores['rougeL'].fmeasure
            }

            rouge_results.append(rouge_result)
            
            # Write the results to a JSON file
            with open(results_path, 'w') as results_file:
                json.dump({"metadata": metadata, "data": rouge_results}, results_file, indent=4)
    
    print("Ending ROUGE Test")

# Haven't checked if these below tests work yet:
# def calculate_f1(predictions, labels):
#     # Assuming your predictions and labels are already binary or multi-class
#     f1 = f1_score(labels, predictions, average='weighted')
#     print(f"F1 Score: {f1}")
#     return f1

# def measure_latency(model, tokenizer, dataset, device, n=100):
#     start_times = []
#     end_times = []

#     for i, example in enumerate(dataset):
#         if i >= n: break  # Limit to n samples
#         question = example['question']

#         start_time = time.time()
#         input_ids = tokenizer.encode(question, return_tensors='pt').to(device)

#         with torch.no_grad():
#             output = model.generate(input_ids, max_length=50)
        
#         end_time = time.time()

#         start_times.append(start_time)
#         end_times.append(end_time)

#     latencies = [end - start for start, end in zip(start_times, end_times)]
#     average_latency = np.mean(latencies)
#     print(f"Average Latency: {average_latency*1000:.2f} ms")
#     return average_latency



print("Starting Testing")
start_time = time.time()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



loading_start = time.time()
# Load the trained model
model_path = "model/20240222-022652"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
loading_end = time.time()
print(f"Model loaded in {loading_end - loading_start:.2f} seconds")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# ChatML Template
tokenizer.chat_template = """
# System Prompt: When responding to user inquiries, always prioritize safety and accuracy. For any health-related question, provide well-informed, general advice based on current medical understanding. If a question pertains to symptoms, conditions, treatments, or any scenario that could potentially be serious or life-threatening, gently encourage the user to seek professional medical advice or emergency services. Emphasize the importance of consulting healthcare professionals for personal medical concerns. Be empathetic, clear, and cautious in your responses, ensuring users understand that this chatbot does not replace professional medical consultation or emergency services.
{% for message in messages %}
{{ message['role'] + ':\n' + message['content'] + '\n' }}
{% endfor %}
"""


df = load_dataset(file_path="medquad.csv", rows = 10000)
train_dataset, val_dataset, test_dataset = split_dataset(df)

print("Loaded Dataset")

# Generate JSON of questions
generate_and_save_questions(750, test_dataset, model, tokenizer, model_name=model_path)

# Tests
calculate_and_compare_perplexity(model, tokenizer, train_dataset, test_dataset)
test_bleu_score("Saved Responses/questions_20240322-030647.json")
test_rouge_score("Saved Responses/questions_20240322-030647.json")

