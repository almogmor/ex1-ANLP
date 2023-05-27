import sys
import os
os.environ['TRANSFORMERS_CACHE'] = '/cs/labs/tomhope/almog.mor/ANLP_ex1/'

import wandb
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel, Trainer, EvalPrediction, TrainingArguments, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import time
import torch

wandb.login(key='31cfe609debc67a554819c517d938d066426d947')


MODELS = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]

def limit_num_of_samples(dataset, num_of_samples):
    if num_of_samples == -1:
        return dataset
    return dataset.select(range(num_of_samples))

def get_dataset(number_of_training_samples, number_of_validation_samples,tokenizer, split='train'):
    dataset = load_dataset('sst2')

    def preprocess_function(examples):
        result = tokenizer(examples['sentence'], padding='max_length', truncation=True)
        return result

    train_set = dataset['train']
    validation_set = dataset['validation']

    limited_train = limit_num_of_samples(train_set, number_of_training_samples)
    limited_validation = limit_num_of_samples(validation_set, number_of_validation_samples)

    tokenized_train = limited_train.map(preprocess_function, batched=True)
    tokenized_validation = limited_validation.map(preprocess_function, batched=True)

    tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_validation.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    return tokenized_train, tokenized_validation

def get_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config, num_labels=2)
    return model, tokenizer

def get_args(args):
    if len(args) != 5:
        print("num of params is invalid")
        return
    return {
        "number_of_seeds": int(sys.argv[1]),
        "number_of_training_samples": int(sys.argv[2]),
        "number_of_validation_samples": int(sys.argv[3]),
        "number_of_prediction_samples": int(sys.argv[4])
    }

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = accuracy_score(preds, p.label_ids)
    return {"accuracy": result}



def train(model_name, seed, number_of_seeds, number_of_training_samples, number_of_validation_samples, number_of_prediction_samples):

    wandb.init(project='ANLP-ex1', name=f'{model_name}_{seed}')

    training_args = TrainingArguments(
        f"train-sentiment_analysis",
        report_to="wandb"
    )


    transformers.set_seed(seed)
    model, tokenizer = get_model(model_name=model_name)
    train_set, validation_set = get_dataset(number_of_training_samples, number_of_validation_samples, tokenizer)
    trainer = Trainer(
              model=model,
              args=training_args,
              train_dataset=train_set,
              eval_dataset=validation_set,
              compute_metrics=compute_metrics,
              tokenizer=tokenizer
          )
    train_result = trainer.train()
    metrics = trainer.evaluate(eval_dataset=validation_set)
    return model, tokenizer, train_result, metrics


def write_predictions(model, tokenizer, num_of_samples):
    dataset = load_dataset('sst2')
    test_set = dataset['test']
    limited_test_set = limit_num_of_samples(test_set, num_of_samples)

    test_set = limited_test_set['sentence']
    prediction_to_write = []
    start_prediction_time = time.time()
    for test_example in test_set:
        print(f"test_example:{test_example}")
        inputs = tokenizer(test_example, truncation=True, max_length=512, return_tensors="pt")

        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits).item()
        prediction_to_write.append(f"{test_example}###{predicted_label}")
    prediction_time = time.time() - start_prediction_time

    with open("predictions.txt", 'w') as f:
        for line in prediction_to_write:
          f.write(f"{line}\n")
    return prediction_time

def main():
    args = get_args(args=sys.argv)
    print(args)

    results = {}
    best_model = ""
    best_tokenizer = ""
    best_acc = 0
    mean_std_results = []
    training_time_start = 0

    for model_name in MODELS:
        training_time_start = time.time()
        accuracy = []
        for seed in range(args.get('number_of_seeds')):
            model,tokenizer, train_result, metrics = train(model_name,seed, **args)
            results[f"{model}_{seed}"] = {"train_result":train_result, "metrics":metrics}
            accuracy.append(metrics['eval_accuracy'])
        accuracy = np.array(accuracy)
        mean_acc, std_acc = np.mean(accuracy), np.std(accuracy)
        mean_std_results.append(f"{model_name},{mean_acc} +- {std_acc}")

        if mean_acc > best_acc:
            best_model = model
            best_tokenizer = tokenizer
            best_acc = mean_acc
    training_time = time.time() - training_time_start
    print(results)
    prediction_time = write_predictions(best_model,best_tokenizer, args['number_of_prediction_samples'])
    wandb.finish()

    with open("res.txt", 'w') as f:
      for results_str in mean_std_results:
        f.write(f"{results_str}\n")
      f.write("----\n")
      f.write(f"train_time,{training_time}\n")
      f.write(f"prediction_time,{prediction_time}")

#python ex1.py <number of seeds> <number of training samples> <number of validation samples> <number of prediction samples>
if __name__ == '__main__':
    main()