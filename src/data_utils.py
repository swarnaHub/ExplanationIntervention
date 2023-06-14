import os
import json
import pandas
import re


class StrategyQA:
    def __init__(self, data_dir):
        self.train_path = os.path.join(data_dir, "train.json")
        self.dev_path = os.path.join(data_dir, "dev.json")
        self.test_path = os.path.join(data_dir, "strategyqa_test.json")

    def get_samples(self, file_path):
        samples = []
        with open(file_path, "r", encoding="utf-8-sig") as f:
            json_inputs = json.load(f)
            for i, json_input in enumerate(json_inputs):
                samples.append({
                    "index": i,
                    "qid": json_input["qid"],
                    "question": json_input["question"],
                    "answer": "yes" if json_input["answer"] else "no",
                    "gold_explanation": " ".join(json_input["facts"])
                })

        return samples

    def get_train_samples(self):
        return self.get_samples(self.train_path)

    def get_dev_samples(self):
        return self.get_samples(self.dev_path)

    def get_test_samples(self):
        return self.get_samples(self.test_path)


class GSM8k:
    def __init__(self, data_dir):
        self.train_path = os.path.join(data_dir, "train.jsonl")
        self.test_path = os.path.join(data_dir, "test.jsonl")

    def get_samples(self, file_path):
        samples = []
        count = 0
        with open(file_path, "r") as f:
            jsonlines = f.read().splitlines()
            for i, jsonline in enumerate(jsonlines):
                sample = json.loads(jsonline)
                answer = re.sub(r"[^0-9.]", "",sample["answer"].split("#### ")[1].strip())
                gold_explanation = re.sub('<<.*>>', '', sample["answer"].split("#### ")[0].replace("\n\n", "\n").strip())
                gold_explanation_sents = gold_explanation.split("\n")
                gold_explanation_sents = [gold_explanation_sent + "." if gold_explanation_sent[-1] != "." else gold_explanation_sent for gold_explanation_sent in gold_explanation_sents]
                gold_explanation = " ".join(gold_explanation_sents)
                sample_json = {
                    "index": i,
                    "question": sample["question"],
                    "answer": answer,
                    "gold_explanation": gold_explanation
                }
                samples.append(sample_json)

        return samples

    def get_train_samples(self):
        return self.get_samples(self.train_path)

    def get_test_samples(self):
        return self.get_samples(self.test_path)


class ECQA:
    def __init__(self, data_dir):
        self.train_path = os.path.join(data_dir, "cqa_data_train.csv")
        self.dev_path = os.path.join(data_dir, "cqa_data_val.csv")
        self.test_path = os.path.join(data_dir, "cqa_data_test.csv")

    def get_samples(self, file_path):
        samples = []
        df = pandas.read_csv(file_path)
        for index, row in df.iterrows():
            options = [row["q_op1"], row["q_op2"], row["q_op3"], row["q_op4"], row["q_op5"]]
            samples.append({
                "index": index,
                "question": row["q_text"],
                "options": options,
                "answer": str(options.index(row["q_ans"]) + 1),
                "gold_explanation": row["taskB"]
            })

        return samples

    def get_train_samples(self):
        return self.get_samples(self.train_path)

    def get_dev_samples(self):
        return self.get_samples(self.dev_path)

    def get_test_samples(self):
        return self.get_samples(self.test_path)
