from torch.nn.functional import softmax
from tqdm import tqdm
import re


class TeacherModel:
    def __init__(self,
                 model_name,
                 model=None,
                 dataset=None,
                 expl_type=None,
                 tokenizer=None,
                 in_context_samples=None,
                 max_new_tokens=100,
                 num_beams=1
                 ):
        self.model_name = model_name
        self.model = model
        self.dataset = dataset
        self.expl_type = expl_type
        self.tokenizer = tokenizer
        self.in_context_samples = in_context_samples
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def set_ics(self, in_context_samples):
        self.in_context_samples = in_context_samples

    def prepare_context_rational(self, test_sample):
        if self.dataset == "strategyQA":
            context = "\n\n".join([f"Q: {in_context_sample['question']}\nA: {in_context_sample['answer']} because {in_context_sample['gold_explanation']}" for in_context_sample in self.in_context_samples])
            context += f"\n\nQ: {test_sample['question']}\nA:"
        elif self.dataset == "ecqa":
            context = "\n\n".join(
                [f"Q: {in_context_sample['question']}\nAnswer Choices:\n" +
                 f"Choice 1: {in_context_sample['options'][0]}\nChoice 2: {in_context_sample['options'][1]}\n" +
                 f"Choice 3: {in_context_sample['options'][2]}\nChoice 4: {in_context_sample['options'][3]}\n" +
                 f"Choice 5: {in_context_sample['options'][4]}\nA: {in_context_sample['answer']} because {in_context_sample['gold_explanation']}"
                 for in_context_sample in self.in_context_samples])
            context = context + f"\n\nQ: {test_sample['question']}\nAnswer Choices:\n" + \
                      f"Choice 1: {test_sample['options'][0]}\nChoice 2: {test_sample['options'][1]}\n" + \
                      f"Choice 3: {test_sample['options'][2]}\nChoice 4: {test_sample['options'][3]}\n" + \
                      f"Choice 5: {test_sample['options'][4]}\nA:"
        else:
            assert False, "Dataset not recognized"
        return context

    def prepare_context_no_expl(self, test_sample):
        if self.dataset == "strategyQA":
            context = "\n\n".join(
              [f"Q: {in_context_sample['question']}\nA: The answer is {in_context_sample['answer']}" for
               in_context_sample
               in self.in_context_samples])
            context += f"\n\nQ: {test_sample['question']}\nA: The answer is"
        elif self.dataset == "ecqa":
            context = "\n\n".join(
              [f"Q: {in_context_sample['question']}\nAnswer Choices:\n" +
               f"Choice 1: {in_context_sample['options'][0]}\nChoice 2: {in_context_sample['options'][1]}\n" +
               f"Choice 3: {in_context_sample['options'][2]}\nChoice 4: {in_context_sample['options'][3]}\n" +
               f"Choice 5: {in_context_sample['options'][4]}\nA: The correct choice is {in_context_sample['answer']}"
               for in_context_sample in self.in_context_samples])
            context = context + f"\n\nQ: {test_sample['question']}\nAnswer Choices:\n" + \
                  f"Choice 1: {test_sample['options'][0]}\nChoice 2: {test_sample['options'][1]}\n" + \
                  f"Choice 3: {test_sample['options'][2]}\nChoice 4: {test_sample['options'][3]}\n" + \
                  f"Choice 5: {test_sample['options'][4]}\nA: The correct choice is"
        else:
            assert False, "Dataset not recognized"
        return context

    def prepare_context_CoT(self, test_sample):
        if self.dataset == "strategyQA":
            context = "\n\n".join([
                f"Q: {in_context_sample['question']}\nA: {in_context_sample['gold_explanation']} So the answer is {in_context_sample['answer']}"
                for in_context_sample in self.in_context_samples])
            context += f"\n\nQ: {test_sample['question']}\nA:"
        elif self.dataset == "ecqa":
            context = "\n\n".join(
                [f"Q: {in_context_sample['question']}\nAnswer Choices:\n" +
                 f"Choice 1: {in_context_sample['options'][0]}\nChoice 2: {in_context_sample['options'][1]}\n" +
                 f"Choice 3: {in_context_sample['options'][2]}\nChoice 4: {in_context_sample['options'][3]}\n" +
                 f"Choice 5: {in_context_sample['options'][4]}\nA: {in_context_sample['gold_explanation']}" +
                 f" So the correct choice is {in_context_sample['answer']}" for in_context_sample in
                 self.in_context_samples])
            context = context + f"\n\nQ: {test_sample['question']}\nAnswer Choices:\n" + \
                      f"Choice 1: {test_sample['options'][0]}\nChoice 2: {test_sample['options'][1]}\n" + \
                      f"Choice 3: {test_sample['options'][2]}\nChoice 4: {test_sample['options'][3]}\n" + \
                      f"Choice 5: {test_sample['options'][4]}\nA:"
        elif self.dataset == "gsm8k":
            context = "\n\n".join([
                f"Q: {in_context_sample['question']}\nA: {in_context_sample['gold_explanation']} So the answer is {in_context_sample['answer']}"
                for in_context_sample in self.in_context_samples])
            context += f"\n\nQ: {test_sample['question']}\nA:"
        else:
            assert False, "Dataset not recognized"

        return context

    def prepare_context_own_explanation(self, test_sample, explanation):
        if self.dataset == "strategyQA":
            context = "\n\n".join([
                f"Q: {in_context_sample['question']}\nA: {in_context_sample['gold_explanation']} So the answer is {in_context_sample['answer']}"
                for in_context_sample in self.in_context_samples])
            context += f"\n\nQ: {test_sample['question']}\nA: {explanation} So the answer is"
        elif self.dataset == "ecqa":
            context = "\n\n".join(
                [f"Q: {in_context_sample['question']}\nAnswer Choices:\n" +
                 f"Choice 1: {in_context_sample['options'][0]}\nChoice 2: {in_context_sample['options'][1]}\n" +
                 f"Choice 3: {in_context_sample['options'][2]}\nChoice 4: {in_context_sample['options'][3]}\n" +
                 f"Choice 5: {in_context_sample['options'][4]}\nA: {in_context_sample['gold_explanation']}" +
                 f" So the correct choice is {in_context_sample['answer']}" for in_context_sample in
                 self.in_context_samples])
            context = context + f"\n\nQ: {test_sample['question']}\nAnswer Choices:\n" + \
                      f"Choice 1: {test_sample['options'][0]}\nChoice 2: {test_sample['options'][1]}\n" + \
                      f"Choice 3: {test_sample['options'][2]}\nChoice 4: {test_sample['options'][3]}\n" + \
                      f"Choice 5: {test_sample['options'][4]}\nA: {explanation} So the correct choice is"
        else:
            assert False, "Dataset not recognized"
        return context

    def predict_single_confidence(self, test_sample, with_expl=False):
        context = self.prepare_context_no_expl(test_sample=test_sample) if not with_expl else self.prepare_context_CoT(test_sample=test_sample)
        tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
        generated = self.model.generate(**tokens, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens, output_scores=True, return_dict_in_generate=True)

        idx = 1 if "llam" in self.model_name else 0
        if self.dataset == "strategyQA":
            yes_id, no_id = self.tokenizer.encode("yes")[idx], self.tokenizer.encode("no")[idx]
            answer_id = 0
            if with_expl:
                generated_tokens = generated[0].squeeze().tolist()
                if yes_id in generated_tokens or no_id in generated_tokens:
                    answer_id = generated_tokens.index(yes_id)-1 if yes_id in generated_tokens else generated_tokens.index(no_id)-1
            scores = softmax(generated['scores'][answer_id], dim=-1)

            yes_score, no_score = scores[0][yes_id].item(), scores[0][no_id].item()
            print(f'Yes score = {yes_score}')
            print(f'No score = {no_score}')
            class_scores = [yes_score, no_score]
        elif self.dataset == "ecqa":
            option1_id, option2_id, option3_id, option4_id, option5_id = self.tokenizer.encode("1")[idx], \
                                                                         self.tokenizer.encode("2")[idx], \
                                                                         self.tokenizer.encode("3")[idx], \
                                                                         self.tokenizer.encode("4")[idx], \
                                                                         self.tokenizer.encode("5")[idx]
            option1_score, option2_score, option3_score, option4_score, option5_score = scores[0][option1_id].item(), \
                                                                                        scores[0][option2_id].item(), \
                                                                                        scores[0][option3_id].item(), \
                                                                                        scores[0][option4_id].item(), \
                                                                                        scores[0][option5_id].item()
            print(f'Option1 score = {option1_score}')
            print(f'Option2 score = {option2_score}')
            print(f'Option3 score = {option3_score}')
            print(f'Option4 score = {option4_score}')
            print(f'Option5 score = {option5_score}')
            class_scores = [option1_score, option2_score, option3_score, option4_score, option5_score]
        else:
            assert False, "Dataset not recognized"

        return class_scores

    def predict_single(self, test_sample):
        if self.model_name == "human":
            return None, test_sample["gold_explanation"]
        else:
            if self.expl_type == "blind_teacher_rationalize":
                context = self.prepare_context_rational(test_sample=test_sample)
            elif self.expl_type == "blind_teacher_CoT" or self.expl_type == "useful_teacher":
                    context = self.prepare_context_CoT(test_sample=test_sample)
            else:
                assert False, "ToM type not supported"
            tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
            generated = self.model.generate(**tokens, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
            output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

        if "llama" in self.model_name:
            output = output[len(context):]
        output = output[:output.index('\n')].strip() if '\n' in output else output.strip()

        if "The correct choice is " in output:
            output = output[len("The correct choice is "):].strip()

        if self.expl_type == "blind_teacher_rationalize":
            if self.dataset == "ecqa":
                if output not in ["1", "2", "3", "4", "5"]:
                    for i, choice in enumerate(test_sample["options"]):
                        if choice in output:
                            output = str(i + 1)
                            break
            prediction = output.split(" ")[0]
            print(f'Teacher Prediction = {prediction}')
            explanation = " ".join(output.split(" ")[2:])
            print(f'Teacher Explanation = {explanation}')
        else:
            explanation = output[:output.rfind(".") + 1]
            print(f'Teacher Explanation = {explanation}')
            prediction = output.split(" ")[-1]
            if self.dataset == "ecqa":
                if prediction not in ["1", "2", "3", "4", "5"]:
                    for i, choice in enumerate(test_sample["options"]):
                        if choice in output:
                            prediction = str(i + 1)
                            break
            elif self.dataset == "strategyQA":
                if prediction not in ["no", "yes"]:
                    print("Regenerating with the explanation")
                    context = self.prepare_context_own_explanation(test_sample, explanation)
                    tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
                    generated = self.model.generate(**tokens, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
                    output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
                    output = output[len(context):] if context in output else output
                    output = output[:output.index('\n')].strip() if '\n' in output else output.strip()
                    prediction = output.split(" ")[0]
                print(f'Teacher Prediction = {prediction}')
            elif self.dataset == "gsm8k":
                prediction = re.sub(r"[^0-9.]", "", prediction)
                if prediction == "" or prediction == ".":
                    for word in reversed(explanation.split(" ")):
                        if bool(re.search(r"\d", word)):
                            prediction = re.sub(r"[^0-9.]", "", word)
                            break

        return prediction, explanation

    def predict(self, test_samples):
        explanations = []
        for test_sample in tqdm(test_samples):
            _, explanation = self.predict_single(test_sample=test_sample)
            explanations.append(explanation)

        return explanations
