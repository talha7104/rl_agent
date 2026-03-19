# generate_data.py
import json
import random

random.seed(42)

# Sentiment classification data
# The model needs to learn: positive/negative sentiment from short reviews
POSITIVE = [
    "This product is excellent and works perfectly.",
    "Amazing quality, highly recommend to everyone.",
    "Best purchase I have made in years.",
    "Outstanding performance, very satisfied.",
    "Great value for money, love it.",
    "Fantastic experience from start to finish.",
    "Superb quality and fast delivery.",
    "Exceeded my expectations completely.",
    "Wonderful product, works like a charm.",
    "Brilliant, exactly what I needed.",
]

NEGATIVE = [
    "This product is terrible and broke immediately.",
    "Awful quality, complete waste of money.",
    "Worst purchase I have ever made.",
    "Disappointing performance, very unsatisfied.",
    "Poor value, totally not worth it.",
    "Horrible experience from start to finish.",
    "Cheap quality and slow delivery.",
    "Did not meet my expectations at all.",
    "Dreadful product, stopped working quickly.",
    "Useless, nothing like what was advertised.",
]

def make_example(text, label):
    instruction = f"Classify the sentiment of this review as positive or negative: {text}"
    response = label
    return {"instruction": instruction, "response": response}

def make_eval_example(text, label):
    prompt = f"Classify the sentiment of this review as positive or negative: {text}"
    return {"prompt": prompt, "label": label}

# Generate train.jsonl — 200 examples
train = []
for _ in range(100):
    train.append(make_example(random.choice(POSITIVE), "positive"))
    train.append(make_example(random.choice(NEGATIVE), "negative"))
random.shuffle(train)

with open("data/train.jsonl", "w") as f:
    for ex in train:
        f.write(json.dumps(ex) + "\n")

# Generate val.jsonl — 40 examples
val = []
for _ in range(20):
    val.append(make_example(random.choice(POSITIVE), "positive"))
    val.append(make_example(random.choice(NEGATIVE), "negative"))
random.shuffle(val)

with open("data/val.jsonl", "w") as f:
    for ex in val:
        f.write(json.dumps(ex) + "\n")

# Generate judge_data/eval.jsonl — 60 examples (hidden from LLM)
eval_data = []
for _ in range(30):
    eval_data.append(make_eval_example(random.choice(POSITIVE), "positive"))
    eval_data.append(make_eval_example(random.choice(NEGATIVE), "negative"))
random.shuffle(eval_data)

with open("judge_data/eval.jsonl", "w") as f:
    for ex in eval_data:
        f.write(json.dumps(ex) + "\n")

print("Data generated successfully.")
print(f"  data/train.jsonl    : {len(train)} examples")
print(f"  data/val.jsonl      : {len(val)} examples")
print(f"  judge_data/eval.jsonl: {len(eval_data)} examples (hidden eval set)")

# Preview one example from each
print("\nTrain example:")
print(json.dumps(train[0], indent=2))
print("\nEval example:")
print(json.dumps(eval_data[0], indent=2))