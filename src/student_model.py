from torch.nn.functional import softmax
from tqdm import tqdm
import re

def compute_accuracy(labels, predictions):
    correct = 0
    for (label, prediction) in zip(labels, predictions):
        if label == prediction:
            correct += 1

    return correct / len(labels)


class StudentModel:
    def __init__(self,
                 model_name,
                 model,
                 dataset,
                 tokenizer,
                 in_context_samples,
                 use_explanations,
                 intervention_strategy,
                 intervention_action,
                 no_intervention_action,
                 max_new_tokens=100,
                 num_beams=1):
        self.model_name = model_name
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.in_context_samples = in_context_samples
        self.use_explanations = use_explanations
        self.intervention_strategy = intervention_strategy
        self.intervention_action = intervention_action
        self.no_intervention_action = no_intervention_action
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def set_ics(self, in_context_samples):
        self.in_context_samples = in_context_samples

    def add_additional_ics(self, additional_ics):
        self.in_context_samples = self.in_context_samples + additional_ics

    def set_intervention_action(self, action):
        self.intervention_action = action

    def prepare_context_no_expl(self, test_sample):
        if self.dataset == "strategyQA":
            context = "\n\n".join(
                [f"Q: {in_context_sample['question']}\nA: The answer is {in_context_sample['answer']}" for in_context_sample in self.in_context_samples])
            context += f"\n\nQ: {test_sample['question']}\nA: The answer is"
        elif self.dataset == "ecqa":
            context = "\n\n".join(
                [f"Q: {in_context_sample['question']}\nAnswer Choices:\n" +
                 f"Choice 1: {in_context_sample['options'][0]}\nChoice 2: {in_context_sample['options'][1]}\n" +
                 f"Choice 3: {in_context_sample['options'][2]}\nChoice 4: {in_context_sample['options'][3]}\n" +
                 f"Choice 5: {in_context_sample['options'][4]}\nA: The correct choice is {in_context_sample['answer']}" for in_context_sample in self.in_context_samples])
            context = context + f"\n\nQ: {test_sample['question']}\nAnswer Choices:\n" + \
                      f"Choice 1: {test_sample['options'][0]}\nChoice 2: {test_sample['options'][1]}\n" + \
                      f"Choice 3: {test_sample['options'][2]}\nChoice 4: {test_sample['options'][3]}\n" + \
                      f"Choice 5: {test_sample['options'][4]}\nA: The correct choice is"
        elif self.dataset == "gsm8k":
            context = "\n\n".join(
                [f"Q: {in_context_sample['question']}\nA: The answer is {in_context_sample['answer']}" for in_context_sample in self.in_context_samples])
            context += f"\n\nQ: {test_sample['question']}\nA: The answer is"
        else:
            assert False, "Dataset not recognized"
        return context

    def prepare_context_CoT(self, test_sample):
        if self.dataset == "strategyQA":
            context = "\n\n".join([
                f"Q: {in_context_sample['question']}\nA: {in_context_sample['gold_explanation']} So the answer is {in_context_sample['answer']}" for in_context_sample in self.in_context_samples])
            context += f"\n\nQ: {test_sample['question']}\nA:"
        elif self.dataset == "ecqa":
            context = "\n\n".join(
                [f"Q: {in_context_sample['question']}\nAnswer Choices:\n" +
                 f"Choice 1: {in_context_sample['options'][0]}\nChoice 2: {in_context_sample['options'][1]}\n" +
                 f"Choice 3: {in_context_sample['options'][2]}\nChoice 4: {in_context_sample['options'][3]}\n" +
                 f"Choice 5: {in_context_sample['options'][4]}\nA: {in_context_sample['gold_explanation']}" +
                 f" So the correct choice is {in_context_sample['answer']}" for in_context_sample in self.in_context_samples])
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

    def prepare_context_teacher_explanation(self, test_sample, teacher_explanation):
        if self.dataset == "strategyQA":
            context = "\n\n".join([
                f"Q: {in_context_sample['question']}\nA: {in_context_sample['gold_explanation']} So the answer is {in_context_sample['answer']}" for in_context_sample in self.in_context_samples])
            context += f"\n\nQ: {test_sample['question']}\nA: {teacher_explanation} So the answer is"
        elif self.dataset == "ecqa":
            context = "\n\n".join(
                [f"Q: {in_context_sample['question']}\nAnswer Choices:\n" +
                 f"Choice 1: {in_context_sample['options'][0]}\nChoice 2: {in_context_sample['options'][1]}\n" +
                 f"Choice 3: {in_context_sample['options'][2]}\nChoice 4: {in_context_sample['options'][3]}\n" +
                 f"Choice 5: {in_context_sample['options'][4]}\nA: {in_context_sample['gold_explanation']}" +
                 f" So the correct choice is {in_context_sample['answer']}" for in_context_sample in self.in_context_samples])
            context = context + f"\n\nQ: {test_sample['question']}\nAnswer Choices:\n" + \
                      f"Choice 1: {test_sample['options'][0]}\nChoice 2: {test_sample['options'][1]}\n" + \
                      f"Choice 3: {test_sample['options'][2]}\nChoice 4: {test_sample['options'][3]}\n" + \
                      f"Choice 5: {test_sample['options'][4]}\nA: {teacher_explanation} So the correct choice is"
        elif self.dataset == "gsm8k":
            context = "\n\n".join([
                f"Q: {in_context_sample['question']}\nA: {in_context_sample['gold_explanation']} So the answer is {in_context_sample['answer']}"
                for in_context_sample in self.in_context_samples])
            test_sample_explanation_sents = teacher_explanation.split(".")
            test_sample_partial_explanation = test_sample_explanation_sents[0] + "."
            print(f"Partial explanation = {test_sample_partial_explanation}")
            context += f"\n\nQ: {test_sample['question']}\nA: {test_sample_partial_explanation}"
        else:
            assert False, "Dataset not recognized"
        return context

    def prepare_context(self, test_sample, inside_mm, intervene, tm):
        if self.use_explanations:
            if intervene:
                _, teacher_explanation = tm.predict_single(test_sample)
                return self.prepare_context_teacher_explanation(test_sample, teacher_explanation)
            else:
                if self.no_intervention_action == "rational":
                    return self.prepare_context_rational(test_sample)
                elif self.no_intervention_action == "CoT":
                    return self.prepare_context_CoT(test_sample)
                elif self.no_intervention_action == "no expl":
                    return self.prepare_context_no_expl(test_sample)
                else:
                    assert False, "No intervention action not defined"
        else:
            return self.prepare_context_no_expl(test_sample)

    def listRightIndex(self, alist, value):
        return len(alist) - alist[-1::-1].index(value) -1

    def predict_single_confidence(self, test_sample, expl=None, with_expl=False):
        if not expl:
            context = self.prepare_context_no_expl(test_sample=test_sample) if not with_expl else self.prepare_context_CoT(test_sample=test_sample)
        else: 
            context = self.prepare_context_teacher_explanation(test_sample=test_sample, teacher_explanation=expl)
        tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
        generated = self.model.generate(**tokens, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens, output_scores=True, return_dict_in_generate=True)
        output = self.tokenizer.batch_decode(generated[0], skip_special_tokens=True)[0].strip()

        if self.dataset == "strategyQA":
            if "llama" in self.model_name:
                yes_id, no_id = self.tokenizer.encode("yes")[1], self.tokenizer.encode("no")[1]
            else:
                yes_id, no_id = self.tokenizer.encode("yes")[0], self.tokenizer.encode("no")[0]

            if with_expl and not expl:
                if "llama" in self.model_name:
                    end_id = self.tokenizer.encode("\n")[2]
                    answer_id = len(tokens["input_ids"][0])
                else:
                    end_id = self.tokenizer.encode("\n")[0]
                    answer_id = 1
            
                generated_tokens = generated[0].squeeze().tolist()[answer_id:]
                if end_id in generated_tokens:
                    generated_tokens = generated_tokens[:generated_tokens.index(end_id)]

                if yes_id in generated_tokens or no_id in generated_tokens:
                    answer_id = generated_tokens.index(yes_id) if yes_id in generated_tokens else generated_tokens.index(no_id)
                else:
                    answer_id = 0

            else:
                answer_id = 0

            scores = softmax(generated['scores'][answer_id], dim=-1)
            yes_score, no_score = scores[0][yes_id].item(), scores[0][no_id].item()
            print(f'Yes score = {yes_score}')
            print(f'No score = {no_score}')
            class_scores = [yes_score, no_score]
        elif self.dataset == "ecqa":
            if "llama" in self.model_name:
                option1_id, option2_id, option3_id, option4_id, option5_id = self.tokenizer.encode("1")[1], \
                                                                             self.tokenizer.encode("2")[1], \
                                                                             self.tokenizer.encode("3")[1], \
                                                                             self.tokenizer.encode("4")[1], \
                                                                             self.tokenizer.encode("5")[1]
                option1_text_id, option2_text_id, option3_text_id, option4_text_id, option5_text_id = \
                                                                             self.tokenizer.encode(test_sample["options"][0].split(" ")[0])[1], \
                                                                             self.tokenizer.encode(test_sample["options"][1].split(" ")[0])[1], \
                                                                             self.tokenizer.encode(test_sample["options"][2].split(" ")[0])[1], \
                                                                             self.tokenizer.encode(test_sample["options"][3].split(" ")[0])[1], \
                                                                             self.tokenizer.encode(test_sample["options"][4].split(" ")[0])[1] 
            else:
                option1_id, option2_id, option3_id, option4_id, option5_id = self.tokenizer.encode("1")[0], \
                                                                             self.tokenizer.encode("2")[0], \
                                                                             self.tokenizer.encode("3")[0], \
                                                                             self.tokenizer.encode("4")[0], \
                                                                             self.tokenizer.encode("5")[0]
                option1_text_id, option2_text_id, option3_text_id, option4_text_id, option5_text_id = \
                                                                             self.tokenizer.encode(test_sample["options"][0].split(" ")[0])[0], \
                                                                             self.tokenizer.encode(test_sample["options"][1].split(" ")[0])[0], \
                                                                             self.tokenizer.encode(test_sample["options"][2].split(" ")[0])[0], \
                                                                             self.tokenizer.encode(test_sample["options"][3].split(" ")[0])[0], \
                                                                             self.tokenizer.encode(test_sample["options"][4].split(" ")[0])[0]

            found_text = False
            if with_expl and not expl:
                if "llam" in self.model_name:
                    end_id = self.tokenizer.encode("\n")[2]
                    answer_id = len(tokens["input_ids"][0])
                else:
                    end_id = self.tokenizer.encode("\n")[0]
                    answer_id = 1

                generated_tokens = generated[0].squeeze().tolist()[answer_id:]
                if end_id in generated_tokens:
                    generated_tokens = generated_tokens[:generated_tokens.index(end_id)]

                if option1_id in generated_tokens:
                    answer_id = self.listRightIndex(generated_tokens, option1_id)
                elif option2_id in generated_tokens:
                    answer_id = self.listRightIndex(generated_tokens, option2_id)
                elif option3_id in generated_tokens:
                    answer_id = self.listRightIndex(generated_tokens, option3_id)
                elif option4_id in generated_tokens:
                    answer_id = self.listRightIndex(generated_tokens, option4_id)
                elif option5_id in generated_tokens:
                    answer_id = self.listRightIndex(generated_tokens, option5_id)
                else:
                    found_text = True
                    if option1_text_id in generated_tokens:
                        answer_id = self.listRightIndex(generated_tokens, option1_text_id)
                    if option2_text_id in generated_tokens:
                        answer_id = max(answer_id, self.listRightIndex(generated_tokens, option2_text_id))
                    if option3_text_id in generated_tokens:
                        answer_id = max(answer_id, self.listRightIndex(generated_tokens, option3_text_id))
                    if option4_text_id in generated_tokens:
                        answer_id = max(answer_id, self.listRightIndex(generated_tokens, option4_text_id))
                    if option5_text_id in generated_tokens:
                        answer_id = max(answer_id, self.listRightIndex(generated_tokens, option5_text_id))
            else:
                answer_id = 0
                if output.split(" ")[0] not in ["1", "2", "3", "4", "5"]:
                    found_text = True

            scores = softmax(generated['scores'][answer_id], dim=-1)

            if found_text:
                option1_score, option2_score, option3_score, option4_score, option5_score = scores[0][option1_text_id].item(), \
                                                                                            scores[0][option2_text_id].item(), \
                                                                                            scores[0][option3_text_id].item(), \
                                                                                            scores[0][option4_text_id].item(), \
                                                                                            scores[0][option5_text_id].item()
            else:
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

    def predict_single(self, test_sample, tm, inside_mm=False, intervene=False):
        context = self.prepare_context(test_sample=test_sample, inside_mm=inside_mm, intervene=intervene, tm=tm)
        tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
        generated = self.model.generate(**tokens, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
        output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

        if "llama" in self.model_name:
            output = output[len(context):]
        output = output[:output.index('\n')].strip() if '\n' in output else output.strip()

        if self.dataset == "ecqa" and "The correct choice is " in output:
            output = output[len("The correct choice is "):].strip()

        if not self.use_explanations or self.no_intervention_action != "CoT":
            if self.dataset == "ecqa":
                if output not in ["1", "2", "3", "4", "5"]:
                    for i, choice in enumerate(test_sample["options"]):
                        if choice in output:
                            output = str(i + 1)
                            break
            prediction = output.split(" ")[0]
            print(f'Student Prediction = {prediction}')
            explanation = " ".join(output.split(" ")[2:])
            print(f'Student Explanation = {explanation}')
        else:
            explanation = output[:output.rfind(".") + 1] if self.dataset != "gsm8k" else output
            print(f'Student Explanation = {explanation}')
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
                    context = self.prepare_context_teacher_explanation(test_sample, explanation)
                    tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
                    generated = self.model.generate(**tokens, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
                    output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
                    output = output[len(context):] if context in output else output
                    output = output[:output.index('\n')].strip() if '\n' in output else output.strip()
                    prediction = output.split(" ")[0]

            elif self.dataset == "gsm8k":
                prediction = re.sub(r"[^0-9.]", "", prediction)
                if prediction == "" or prediction == ".":
                    for word in reversed(explanation.split(" ")):
                        if bool(re.search(r"\d", word)):
                            prediction = re.sub(r"[^0-9.]", "", word)
                            break

            print(f'Student Prediction = {prediction}')

        return prediction, explanation

    def predict(self, test_samples, intervention_samples_per_budget, tm):
        questions, labels, predictions_per_budget, explanations_per_budget = [], [], [[] for i in range(len(intervention_samples_per_budget))], [[] for i in range(len(intervention_samples_per_budget))]

        for test_index, test_sample in enumerate(tqdm(test_samples)):
            print("Using student explanation")
            prediction_student_expl, explanation_student = self.predict_single(test_sample=test_sample, tm=tm, intervene=False)

            print("Using teacher explanation")
            # This is not actually explanation teacher, but don't care for final student evaluation
            prediction_teacher_expl, explanation_teacher = self.predict_single(test_sample=test_sample, tm=tm, intervene=True)

            for i, intervention_samples in enumerate(intervention_samples_per_budget):
                if test_index in intervention_samples:
                    predictions_per_budget[i].append(prediction_teacher_expl)
                    explanations_per_budget[i].append(explanation_teacher)
                else:
                    predictions_per_budget[i].append(prediction_student_expl)
                    explanations_per_budget[i].append(explanation_student)

            questions.append(test_sample['question'])
            labels.append(test_sample['answer'])

        return questions, labels, predictions_per_budget, explanations_per_budget
