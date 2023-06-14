from torch.nn.functional import softmax
from tqdm import tqdm


class MentalModel:
    def __init__(self,
                 model_name,
                 model,
                 dataset,
                 tokenizer,
                 no_inter_ics,
                 inter_ics,
                 tm,
                 intervention_strategy,
                 use_gold_label,
                 max_new_tokens_no_inter=100,
                 max_new_tokens_inter=5,
                 num_beams=1):
        self.model_name = model_name
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.no_inter_ics = no_inter_ics
        self.inter_ics = inter_ics
        self.tm = tm
        self.intervention_strategy = intervention_strategy
        self.use_gold_label = use_gold_label
        self.max_new_tokens_no_inter = max_new_tokens_no_inter
        self.max_new_tokens_inter = max_new_tokens_inter
        self.num_beams = num_beams

    def set_ics(self, no_inter_ics, inter_ics):
        self.no_inter_ics = no_inter_ics
        self.inter_ics = inter_ics

    def predict_util_inter(self, prompt, test_sample):
        tokens = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        generated = self.model.generate(**tokens, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens_inter, output_scores=True, return_dict_in_generate=True)
        scores = softmax(generated['scores'][0], dim=-1)
        output = self.tokenizer.batch_decode(generated['sequences'], skip_special_tokens=True)[0].strip()
        if "llama" in self.model_name:
            output = output[len(prompt):]
        output = output[:output.index('\n')].strip() if '\n' in output else output.strip()

        idx = 1 if "llama" in self.model_name else 0
        if self.dataset == "strategyQA":
            yes_id, no_id = self.tokenizer.encode("yes")[idx], self.tokenizer.encode("no")[idx]
            yes_score, no_score = scores[0][yes_id].item(), scores[0][no_id].item()
            print(f'Yes score = {yes_score}')
            print(f'No score = {no_score}')
            option_scores = [yes_score, no_score]
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

            if output not in ["1", "2", "3", "4", "5"]:
                option1_text_id, option2_text_id, option3_text_id, option4_text_id, option5_text_id = \
                                                                                 self.tokenizer.encode(test_sample["options"][0].split(" ")[0])[idx], \
                                                                                 self.tokenizer.encode(test_sample["options"][1].split(" ")[0])[idx], \
                                                                                 self.tokenizer.encode(test_sample["options"][2].split(" ")[0])[idx], \
                                                                                 self.tokenizer.encode(test_sample["options"][3].split(" ")[0])[idx], \
                                                                                 self.tokenizer.encode(test_sample["options"][4].split(" ")[0])[idx] 


                option1_score, option2_score, option3_score, option4_score, option5_score = scores[0][option1_text_id].item(), \
                                                                                            scores[0][option2_text_id].item(), \
                                                                                            scores[0][option3_text_id].item(), \
                                                                                            scores[0][option4_text_id].item(), \
                                                                                            scores[0][option5_text_id].item()

            print(f'Option1 score = {option1_score}')
            print(f'Option2 score = {option2_score}')
            print(f'Option3 score = {option3_score}')
            print(f'Option4 score = {option4_score}')
            print(f'Option5 score = {option5_score}')

            option_scores = [option1_score, option2_score, option3_score, option4_score, option5_score]
        elif self.dataset == "gsm8k":
            output_except_answer = " ".join(output.split(" ")[:-1])
            output_except_answer_tokens = self.tokenizer.encode(output_except_answer)
            answer_start_id = len(output_except_answer_tokens)

            digits = len(test_sample["answer"])
            answer_ids = self.tokenizer.encode(test_sample["answer"])
            assert len(answer_ids) == digits + 2
            answer_score = 0.
            for i, answer_id in enumerate(answer_ids[2:]):
                if answer_start_id+i < len(generated['scores']):
                    scores = softmax(generated['scores'][answer_start_id+i], dim=-1)
                    answer_score += scores[0][answer_id].item()
            answer_score = answer_score / digits
            option_scores = [answer_score]
        return option_scores, output


    def predict_util_no_inter(self, prompt, test_sample):
        tokens = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        generated = self.model.generate(**tokens, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens_inter, output_scores=True, return_dict_in_generate=True)
        scores = softmax(generated['scores'][0], dim=-1)
        output = self.tokenizer.batch_decode(generated['sequences'], skip_special_tokens=True)[0].strip()
        if "llama" in self.model_name:
            output = output[len(prompt):]
        output = output[:output.index('\n')].strip() if '\n' in output else output.strip()

        idx = 1 if "llama" in self.model_name else 0
        if self.dataset == "strategyQA":
            yes_id, no_id = self.tokenizer.encode("yes")[idx], self.tokenizer.encode("no")[idx]
            yes_score, no_score = scores[0][yes_id].item(), scores[0][no_id].item()
            print(f'Yes score = {yes_score}')
            print(f'No score = {no_score}')
            option_scores = [yes_score, no_score]
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

            if output not in ["1", "2", "3", "4", "5"]:
                option1_text_id, option2_text_id, option3_text_id, option4_text_id, option5_text_id = \
                                                                                 self.tokenizer.encode(test_sample["options"][0].split(" ")[0])[idx], \
                                                                                 self.tokenizer.encode(test_sample["options"][1].split(" ")[0])[idx], \
                                                                                 self.tokenizer.encode(test_sample["options"][2].split(" ")[0])[idx], \
                                                                                 self.tokenizer.encode(test_sample["options"][3].split(" ")[0])[idx], \
                                                                                 self.tokenizer.encode(test_sample["options"][4].split(" ")[0])[idx] 


                option1_score, option2_score, option3_score, option4_score, option5_score = scores[0][option1_text_id].item(), \
                                                                                            scores[0][option2_text_id].item(), \
                                                                                            scores[0][option3_text_id].item(), \
                                                                                            scores[0][option4_text_id].item(), \
                                                                                            scores[0][option5_text_id].item()

            print(f'Option1 score = {option1_score}')
            print(f'Option2 score = {option2_score}')
            print(f'Option3 score = {option3_score}')
            print(f'Option4 score = {option4_score}')
            print(f'Option5 score = {option5_score}')

            option_scores = [option1_score, option2_score, option3_score, option4_score, option5_score]
        elif self.dataset == "gsm8k":
            digits = len(test_sample["answer"])
            answer_ids = self.tokenizer.encode(test_sample["answer"])
            assert len(answer_ids) == digits + 2
            answer_score = 0.
            for i, answer_id in enumerate(answer_ids[2:]):
                scores = softmax(generated['scores'][i+1], dim=-1)
                answer_score += scores[0][answer_id].item()
            answer_score = answer_score / digits
            option_scores = [answer_score]

        return option_scores, output

    def prepare_prompt_no_inter(self, test_sample):
        context = "Simulate an AI model's answer for the given question.\n\n"
        if self.dataset == "strategyQA":
            if not self.use_gold_label:
                context += "\n\n".join(
                    [f"Q: {no_inter_ic['question']}\nAI Predicted Answer: {no_inter_ic['prediction']}" for no_inter_ic in self.no_inter_ics])
                context += f"\n\nQ: {test_sample['question']}\nAI Predicted Answer:"
            else:
                context += "\n\n".join(
                    [f"Q: {no_inter_ic['question']}\nCorrect Answer: {no_inter_ic['answer']}\nAI Predicted Answer: {no_inter_ic['prediction']}" for no_inter_ic in self.no_inter_ics])
                context += f"\n\nQ: {test_sample['question']}\nCorrect Answer: {test_sample['answer']}\nAI Predicted Answer:"
        elif self.dataset == "ecqa":
            if not self.use_gold_label:
                context += "\n\n".join(
                    [f"Q: {no_inter_ic['question']}\nAnswer Choices:\n" +
                     f"Choice 1: {no_inter_ic['options'][0]}\nChoice 2: {no_inter_ic['options'][1]}\n" +
                     f"Choice 3: {no_inter_ic['options'][2]}\nChoice 4: {no_inter_ic['options'][3]}\n" +
                     f"Choice 5: {no_inter_ic['options'][4]}\nAI Predicted Answer: {no_inter_ic['prediction']}" for no_inter_ic in self.no_inter_ics])
                context += f"\n\nQ: {test_sample['question']}\nChoice 1: {test_sample['options'][0]}\nChoice 2: {test_sample['options'][1]}\nChoice 3: {test_sample['options'][2]}\nChoice 4: {test_sample['options'][3]}\nChoice 5: {test_sample['options'][4]}\nAI Predicted Answer:"
            else:
                context += "\n\n".join(
                    [f"Q: {no_inter_ic['question']}\nAnswer Choices:\n" +
                     f"Choice 1: {no_inter_ic['options'][0]}\nChoice 2: {no_inter_ic['options'][1]}\n" +
                     f"Choice 3: {no_inter_ic['options'][2]}\nChoice 4: {no_inter_ic['options'][3]}\n" +
                     f"Choice 5: {no_inter_ic['options'][4]}\nAI Predicted Answer: {no_inter_ic['prediction']}" for no_inter_ic in self.no_inter_ics])
                context += f"\n\nQ: {test_sample['question']}\nChoice 1: {test_sample['options'][0]}\nChoice 2: {test_sample['options'][1]}\nChoice 3: {test_sample['options'][2]}\nChoice 4: {test_sample['options'][3]}\nChoice 5: {test_sample['options'][4]}\nAI Predicted Answer:"
        elif self.dataset == "gsm8k":
            context = "\n\n".join([
                f"Q: {no_inter_ic['question']}\nAI Predicted Answer: {no_inter_ic['answer']}" for no_inter_ic in self.no_inter_ics])
            context += f"\n\nQ: {test_sample['question']}\nAI Predicted Answer:"
        else:
            assert False, "Dataset not recognized"

        return context

    def prepare_prompt_inter(self, test_sample):
        _, teacher_explanation = self.tm.predict_single(test_sample)
        print(f'Teacher explanation = {teacher_explanation}')
        context = "Simulate an AI model's answer for the given question.\n\n"
        if self.dataset == "strategyQA":
            if not self.use_gold_label:
                context += "\n\n".join([f"Q: {inter_ic['question']}\nCorrect Answer: {inter_ic['answer']}\nAI Predicted Answer: {inter_ic['teacher_explanation']} So the answer is {inter_ic['prediction']}" for inter_ic in self.inter_ics])
                context += f"\n\nQ: {test_sample['question']}\nCorrect Answer: {test_sample['answer']}\nAI Predicted Answer: {teacher_explanation} So the answer is"
            else:
                _, teacher_explanation = self.tm.predict_single(test_sample)
                context += "\n\n".join([f"Q: {inter_ic['question']}\nCorrect Answer: {inter_ic['answer']}\nAI Predicted Answer: {inter_ic['teacher_explanation']} So the answer is {inter_ic['prediction']}" for inter_ic in self.inter_ics])
                context += f"\n\nQ: {test_sample['question']}\nCorrect Answer: {test_sample['answer']}\nAI Predicted Answer: {teacher_explanation} So the answer is"
        elif self.dataset == "ecqa":
            if not self.use_gold_label:
                context += "\n\n".join(
                    [f"Q: {inter_ic['question']}\nAnswer Choices:\n" +
                     f"Choice 1: {inter_ic['options'][0]}\nChoice 2: {inter_ic['options'][1]}\n" +
                     f"Choice 3: {inter_ic['options'][2]}\nChoice 4: {inter_ic['options'][3]}\n" +
                     f"Choice 5: {inter_ic['options'][4]}\nAI Predicted Answer: {inter_ic['teacher_explanation']} So the correct choice is {inter_ic['prediction']}"
                     for inter_ic in self.inter_ics])
                context += f"\n\nQ: {test_sample['question']}\nChoice 1: {test_sample['options'][0]}\nChoice 2: {test_sample['options'][1]}\nChoice 3: {test_sample['options'][2]}\nChoice 4: {test_sample['options'][3]}\nChoice 5: {test_sample['options'][4]}\nAI Predicted Answer: {teacher_explanation} So the correct choice is"
            else:
                context += "\n\n".join(
                    [f"Q: {inter_ic['question']}\nAnswer Choices:\n" +
                     f"Choice 1: {inter_ic['options'][0]}\nChoice 2: {inter_ic['options'][1]}\n" +
                     f"Choice 3: {inter_ic['options'][2]}\nChoice 4: {inter_ic['options'][3]}\n" +
                     f"Choice 5: {inter_ic['options'][4]}\nCorrect Answer: {inter_ic['answer']}\nAI Predicted Answer: {inter_ic['teacher_explanation']} So the correct choice is {inter_ic['prediction']}"
                     for inter_ic in self.inter_ics])
                context += f"\n\nQ: {test_sample['question']}\nChoice 1: {test_sample['options'][0]}\nChoice 2: {test_sample['options'][1]}\nChoice 3: {test_sample['options'][2]}\nChoice 4: {test_sample['options'][3]}\nChoice 5: {test_sample['options'][4]}\nCorrect Answer: {test_sample['answer']}\nAI Predicted Answer: {teacher_explanation} So the correct choice is"
        elif self.dataset == "gsm8k":
            teacher_explanation_sents = teacher_explanation.split(".")
            teacher_partial_explanation = teacher_explanation_sents[0] + "."
            context = "\n\n".join([f"Q: {inter_ic['question']}\nAI Predicted Answer: {inter_ic['gold_explanation']} So the answer is {inter_ic['answer']}" for inter_ic in self.inter_ics])
            context += f"\n\nQ: {test_sample['question']}\nAI Predicted Answer: {teacher_partial_explanation}"
        else:
            assert False, "Dataset not recognized"

        return context

    def predict(self, test_samples):
        no_inter_predictions, inter_predictions = [], []
        no_inter_correct_scores, inter_correct_scores = [], []
        for test_sample in tqdm(test_samples):
            if self.use_gold_label:
                gold_label = test_sample["answer"]
            else:
                teacher_prediction, _ = self.tm.predict_single(test_sample)
                gold_label = teacher_prediction

            if self.intervention_strategy == "mm_no_inter":
                no_inter_prompt = self.prepare_prompt_no_inter(test_sample)
                option_scores, output = self.predict_util_no_inter(no_inter_prompt, test_sample)

                print(f'AI simulated answer with no intervention (Mental Model) = {output}')

                no_inter_predictions.append(output)
                if self.dataset == "strategyQA":
                    if gold_label == "yes":
                        no_inter_correct_scores.append(option_scores[0])
                    else:
                        no_inter_correct_scores.append(option_scores[1])
                elif self.dataset == "ecqa":
                    if gold_label == "1":
                        no_inter_correct_scores.append(option_scores[0])
                    elif gold_label == "2":
                        no_inter_correct_scores.append(option_scores[1])
                    elif gold_label == "3":
                        no_inter_correct_scores.append(option_scores[2])
                    elif gold_label == "4":
                        no_inter_correct_scores.append(option_scores[3])
                    else:
                        no_inter_correct_scores.append(option_scores[4])
                elif self.dataset == "gsm8k":
                    no_inter_correct_scores.append(option_scores[0])

            elif self.intervention_strategy == "mm_inter":
                inter_prompt = self.prepare_prompt_inter(test_sample)
                option_scores, output = self.predict_util_inter(inter_prompt, test_sample)

                print(f'AI simulated with teacher intervention (Mental Model) = {output}')

                inter_predictions.append(output)
                if self.dataset == "strategyQA":
                    if gold_label == "yes":
                        inter_correct_scores.append(option_scores[0])
                    else:
                        inter_correct_scores.append(option_scores[1])
                elif self.dataset == "ecqa":
                    if gold_label == "1":
                        inter_correct_scores.append(option_scores[0])
                    elif gold_label == "2":
                        inter_correct_scores.append(option_scores[1])
                    elif gold_label == "3":
                        inter_correct_scores.append(option_scores[2])
                    elif gold_label == "4":
                        inter_correct_scores.append(option_scores[3])
                    else:
                        inter_correct_scores.append(option_scores[4])
                elif self.dataset == "gsm8k":
                    inter_correct_scores.append(option_scores[0])

            elif self.intervention_strategy == "mm_both":
                no_inter_prompt = self.prepare_prompt_no_inter(test_sample)
                no_inter_option_scores, no_inter_output = self.predict_util_no_inter(no_inter_prompt, test_sample)

                print(f'AI simulated answer with no intervention (Mental Model) = {no_inter_output}')

                inter_prompt = self.prepare_prompt_inter(test_sample)
                inter_option_scores, inter_output = self.predict_util_inter(inter_prompt, test_sample)

                print(f'AI simulated answer with teacher intervention (Mental Model) = {inter_output}')

                no_inter_predictions.append(no_inter_output)
                inter_predictions.append(inter_output)

                if self.dataset == "strategyQA":
                    if gold_label == "yes":
                        no_inter_correct_scores.append(no_inter_option_scores[0])
                        inter_correct_scores.append(inter_option_scores[0])
                    else:
                        no_inter_correct_scores.append(no_inter_option_scores[1])
                        inter_correct_scores.append(inter_option_scores[1])
                elif self.dataset == "ecqa":
                    if gold_label == "1":
                        no_inter_correct_scores.append(no_inter_option_scores[0])
                        inter_correct_scores.append(inter_option_scores[0])
                    elif gold_label == "2":
                        no_inter_correct_scores.append(no_inter_option_scores[1])
                        inter_correct_scores.append(inter_option_scores[1])
                    elif gold_label == "3":
                        no_inter_correct_scores.append(no_inter_option_scores[2])
                        inter_correct_scores.append(inter_option_scores[2])
                    elif gold_label == "4":
                        no_inter_correct_scores.append(no_inter_option_scores[3])
                        inter_correct_scores.append(inter_option_scores[3])
                    else:
                        no_inter_correct_scores.append(no_inter_option_scores[4])
                        inter_correct_scores.append(inter_option_scores[4])
                elif self.dataset == "gsm8k":
                    no_inter_correct_scores.append(no_inter_option_scores[0])
                    inter_correct_scores.append(inter_option_scores[0])

        return no_inter_predictions, inter_predictions, no_inter_correct_scores, inter_correct_scores
