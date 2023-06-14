import argparse
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import random
import os


from data_utils import StrategyQA, GSM8k, ECQA
from mental_model import MentalModel
from student_model import StudentModel, compute_accuracy
from teacher_model import TeacherModel


def choose_in_context_samples_for_mental_model(args, sm, tm, train_samples):
    print("Choosing in-context samples")
    random.shuffle(train_samples)
    if args.dataset == "strategyQA":
        pre_in_context_samples_yes, pre_in_context_samples_no, post_in_context_samples_yes, post_in_context_samples_no = [], [], [], []
        for train_sample in train_samples:
            if args.intervention_strategy == "mm_no_inter" or args.intervention_strategy == "mm_both":
                if len(pre_in_context_samples_yes) + len(pre_in_context_samples_no) == args.ic_num:
                    break
                student_prediction, student_explanation = sm.predict_single(test_sample=train_sample, tm=tm, inside_mm=False, intervene=False)
                if student_prediction == "yes" and len(pre_in_context_samples_yes) == args.ic_num // 2:
                    continue
                if student_prediction == "no" and len(pre_in_context_samples_no) == args.ic_num // 2:
                    continue
                pre_in_context_sample = {
                    "question": train_sample["question"],
                    "answer": train_sample["answer"],
                    "gold_explanation": train_sample["gold_explanation"],
                    "prediction": student_prediction,
                    "student_explanation": student_explanation
                }

                if student_prediction == "yes":
                    pre_in_context_samples_yes.append(pre_in_context_sample)
                    print("Added one pre yes example")
                else:
                    pre_in_context_samples_no.append(pre_in_context_sample)
                    print("Added one pre no example")

            if args.intervention_strategy == "mm_inter" or args.intervention_strategy == "mm_both":
                if len(post_in_context_samples_yes) + len(post_in_context_samples_no) == args.ic_num:
                    break
                _, teacher_explanation = tm.predict_single(test_sample=train_sample)

                student_prediction, _ = sm.predict_single(test_sample=train_sample, tm=tm, inside_mm=False, intervene=True)
                if student_prediction == "yes" and len(post_in_context_samples_yes) == args.ic_num // 2:
                    continue
                if student_prediction == "no" and len(post_in_context_samples_no) == args.ic_num // 2:
                    continue
                post_in_context_sample = {
                    "question": train_sample["question"],
                    "answer": train_sample["answer"],
                    "gold_explanation": train_sample["gold_explanation"],
                    "prediction": student_prediction,
                    "teacher_explanation": teacher_explanation
                }

                if student_prediction == "yes":
                    post_in_context_samples_yes.append(post_in_context_sample)
                    print("Added one post yes example")
                else:
                    post_in_context_samples_no.append(post_in_context_sample)
                    print("Added one post no example")

        pre_in_context_samples = pre_in_context_samples_yes + pre_in_context_samples_no
        post_in_context_samples = post_in_context_samples_yes + post_in_context_samples_no

        random.shuffle(pre_in_context_samples)
        random.shuffle(post_in_context_samples)
        print("Done")

        return pre_in_context_samples, post_in_context_samples
    elif args.dataset == "ecqa":
        pre_in_context_samples, post_in_context_samples = [], []
        for train_sample in train_samples:
            if args.intervention_strategy == "mm_no_inter" or args.intervention_strategy == "mm_both":
                if len(pre_in_context_samples) == args.ic_num:
                    break
                student_prediction, student_explanation = sm.predict_single(test_sample=train_sample, tm=tm, inside_mm=False, intervene=False)
                pre_in_context_sample = {
                    "question": train_sample["question"],
                    "answer": train_sample["answer"],
                    "options": train_sample["options"],
                    "gold_explanation": train_sample["gold_explanation"],
                    "prediction": student_prediction,
                    "student_explanation": student_explanation
                }
                pre_in_context_samples.append(pre_in_context_sample)

            if args.intervention_strategy == "mm_inter" or args.intervention_strategy == "mm_both":
                if len(post_in_context_samples) == args.ic_num:
                    break
                # This simulates the student post-intervention i.e., with the teacher's explanation
                _, teacher_explanation = tm.predict_single(test_sample=train_sample)
                student_prediction, _ = sm.predict_single(test_sample=train_sample, tm=tm, inside_mm=False, intervene=True)
                post_in_context_sample = {
                    "question": train_sample["question"],
                    "answer": train_sample["answer"],
                    "options": train_sample["options"],
                    "gold_explanation": train_sample["gold_explanation"],
                    "prediction": student_prediction,
                    "teacher_explanation": teacher_explanation
                }
                post_in_context_samples.append(post_in_context_sample)

        random.shuffle(pre_in_context_samples)
        random.shuffle(post_in_context_samples)

        print(pre_in_context_samples)
        print(post_in_context_samples)

        print("Done")
    else:
        pre_in_context_samples, post_in_context_samples = [], []
        for train_sample in train_samples:
            if args.intervention_strategy == "mm_no_inter" or args.intervention_strategy == "mm_both":
                if len(pre_in_context_samples) == args.ic_num:
                    break
                # This simulates the student pre-intervention i.e., with its own explanation
                student_prediction, student_explanation = sm.predict_single(test_sample=train_sample, tm=tm, inside_mm=False, intervene=False)
                pre_in_context_sample = {
                    "question": train_sample["question"],
                    "answer": train_sample["answer"],
                    "gold_explanation": train_sample["gold_explanation"],
                    "prediction": student_prediction,
                    "student_explanation": student_explanation
                }
                pre_in_context_samples.append(pre_in_context_sample)

            if args.intervention_strategy == "mm_inter" or args.intervention_strategy == "mm_both":
                if len(post_in_context_samples) == args.ic_num:
                    break
                # This simulates the student post-intervention i.e., with the teacher's explanation
                _, teacher_explanation = tm.predict_single(test_sample=train_sample)
                student_prediction, _ = sm.predict_single(test_sample=train_sample, tm=tm, inside_mm=False,
                                                          intervene=True)
                post_in_context_sample = {
                    "question": train_sample["question"],
                    "answer": train_sample["answer"],
                    "gold_explanation": train_sample["gold_explanation"],
                    "prediction": student_prediction,
                    "teacher_explanation": teacher_explanation
                }
                post_in_context_samples.append(post_in_context_sample)

        random.shuffle(pre_in_context_samples)
        random.shuffle(post_in_context_samples)

        print(pre_in_context_samples)
        print(post_in_context_samples)

        print("Done")

        return pre_in_context_samples, post_in_context_samples

def choose_in_context_samples_for_teacher_model(args, student_ics, train_samples, sm, tm):
    # Now that the model is loaded, we'll prepare the in-context examples for teacher prompt based on various configurations
    teacher_ics = []
    if args.teacher_expl_type == "blind_teacher_CoT" or args.teacher_expl_type == "blind_teacher_rationalize":
        teacher_ics = student_ics

    # If useful-teacher, then the teacher should generate explanations prompted by "useful" human explanations
    elif args.teacher_expl_type == "useful_teacher":
        random.shuffle(train_samples)
        for train_sample in train_samples:
            student_prediction_pre, _ = sm.predict_single(test_sample=train_sample, tm=tm, inside_mm=False, intervene=False)
            student_prediction_post, _ = sm.predict_single(test_sample=train_sample, tm=tm, inside_mm=False, intervene=True)

            if student_prediction_post == train_sample["answer"] and student_prediction_pre != student_prediction_post:
                teacher_ics.append(train_sample)
                if len(teacher_ics) == args.ic_num:
                    break
    else:
        assert False, "Teacher ToM type not recognized"

    return teacher_ics

def load_all_models(args, train_samples):
    student_in_context_samples = random.sample(train_samples, args.ic_num)
    print("Loading student model!!!")

    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_path, cache_dir=args.cache_dir, use_fast=False)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path, cache_dir=args.cache_dir, use_fast=False) if args.teacher_model_path != "human" else None

    if "llama" in args.student_model_path:
        smodel = AutoModelForCausalLM.from_pretrained(args.student_model_path, cache_dir=args.cache_dir, device_map="auto", torch_dtype=torch.float16)
    else:
        smodel = AutoModelForSeq2SeqLM.from_pretrained(args.student_model_path, device_map="auto", cache_dir=args.cache_dir)

    print("Done")

    sm = StudentModel(model_name=args.student_model_path,
                      model=smodel,
                      dataset=args.dataset,
                      tokenizer=student_tokenizer,
                      in_context_samples=student_in_context_samples,
                      use_explanations=args.use_explanations,
                      intervention_strategy=args.intervention_strategy,
                      intervention_action=args.intervention_action,
                      no_intervention_action=args.no_intervention_action,
                      max_new_tokens=args.max_new_tokens_sm
                      )

    print("Loading teacher model!!!")
    if args.use_explanations:
        # Initially set teacher to human so that when choosing demonstrations for model teacher, the explanations can be human explanations
        tm = TeacherModel(model_name="human")

        if args.teacher_model_path != "human":
            if "llama" in args.student_model_path:
                tmodel = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, cache_dir=args.cache_dir, device_map="auto", torch_dtype=torch.float16)
            else:
                tmodel = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_model_path, device_map="auto", cache_dir=args.cache_dir)

            if args.teacher_expl_type != "human_teacher":
                teacher_ics = choose_in_context_samples_for_teacher_model(args, student_in_context_samples, train_samples, sm, tm)

                tm = TeacherModel(model_name=args.teacher_model_path,
                                model=tmodel,
                                dataset=args.dataset,
                                expl_type=args.teacher_expl_type,
                                tokenizer=teacher_tokenizer,
                                in_context_samples=teacher_ics,
                                max_new_tokens=args.max_new_tokens_tm)			

    else:
        return sm, None, None

    if args.use_explanations and args.intervention_strategy.startswith("mm"):
        print("Loading mental model!!!")
        pre_ics, post_ics = choose_in_context_samples_for_mental_model(args, sm, tm, train_samples)

        # The mental model is basically part of the teacher model.
        mm = MentalModel(model_name=args.mental_model_path,
                         model=tmodel,
                         dataset=args.dataset,
                         tokenizer=teacher_tokenizer,
                         no_inter_ics=pre_ics,
                         inter_ics=post_ics,
                         tm=tm,
                         intervention_strategy=args.intervention_strategy,
                         use_gold_label=args.use_gold_label,
                         max_new_tokens_no_inter=args.max_new_tokens_mm_no_inter,
                         max_new_tokens_inter=args.max_new_tokens_mm_inter)
    else:
        mm = None

    print("Done")
    return sm, tm, mm


def get_intervention_samples(args, mm, sm, tm, budgets, test_samples):
    intervention_samples_per_budget = []
    if args.use_explanations:
        # Random Intervention
        if args.intervention_strategy == "random":
            for budget in budgets:
                budget_count = int(budget * len(test_samples))
                intervention_samples = random.sample(range(0, len(test_samples)), budget_count)
                intervention_samples_per_budget.append(intervention_samples)
        
        # Oracle accuracy
        elif args.intervention_strategy == "oracle":
            intervention_samples = []
            for i, test_sample in enumerate(test_samples):
                student_prediction, _ = sm.predict_single(test_sample=test_sample, tm=tm, inside_mm=False, intervene=False)
                if student_prediction != test_sample["answer"]:
                    intervention_samples.append(i)
            for budget in budgets:
                intervention_samples_per_budget.append(intervention_samples)
        
        # Intervene by ranking with true student confidence
        elif args.intervention_strategy.endswith("student_confidence"):
            sample_confidence_pairs = []
            for i, test_sample in enumerate(test_samples):
                if args.intervention_strategy == "no_intervention_correct_student_confidence":
                    class_scores = sm.predict_single_confidence(test_sample, expl=None, with_expl=True)
                    if args.dataset == "strategyQA":
                        if test_sample["answer"] == "yes":
                            sample_confidence_pairs.append((i, class_scores[0])) 
                        else: 
                            sample_confidence_pairs.append((i, class_scores[1]))
                    elif args.dataset == "ecqa":
                        sample_confidence_pairs.append((i, class_scores[int(test_sample["answer"]) - 1]))
                elif args.intervention_strategy == "least_student_confidence":
                    class_scores = sm.predict_single_confidence(test_sample, expl=None, with_expl=True)
                    sample_confidence_pairs.append((i, min(class_scores)))
                elif args.intervention_strategy == "intervention_correct_student_confidence":
                    _, teacher_explanation = tm.predict_single(test_sample)
                    class_scores = sm.predict_single_confidence(test_sample, expl=teacher_explanation, with_expl=True)
                    if args.dataset == "strategyQA":
                        if test_sample["answer"] == "yes":
                            sample_confidence_pairs.append((i, class_scores[0])) 
                        else: 
                            sample_confidence_pairs.append((i, class_scores[1]))
                    elif args.dataset == "ecqa":
                        sample_confidence_pairs.append((i, class_scores[int(test_sample["answer"]) - 1]))
                elif args.intervention_strategy == "utility_correct_student_confidence":
                    _, teacher_explanation = tm.predict_single(test_sample)
                    with_intervention_class_scores = sm.predict_single_confidence(test_sample, expl=teacher_explanation, with_expl=True)
                    no_intervention_class_scores = sm.predict_single_confidence(test_sample, expl=None, with_expl=True)
                    if args.dataset == "strategyQA":
                        if test_sample["answer"] == "yes":
                            sample_confidence_pairs.append((i, (with_intervention_class_scores[0]-no_intervention_class_scores[0])))
                        else: 
                            sample_confidence_pairs.append((i, (with_intervention_class_scores[1]-no_intervention_class_scores[1])))
                    elif args.dataset == "ecqa":
                        sample_confidence_pairs.append((i, (with_intervention_class_scores[int(test_sample["answer"]) - 1]-no_intervention_class_scores[int(test_sample["answer"]) - 1])))
                else:
                    assert False, "Intervention strategy not supported"
            if args.intervention_strategy == "intervention_correct_student_confidence" or args.intervention_strategy == "utility_correct_student_confidence":
                sample_confidence_pairs = sorted(sample_confidence_pairs, key=lambda x: x[1], reverse=True)
            else:
                sample_confidence_pairs = sorted(sample_confidence_pairs, key=lambda x: x[1])

            for budget in budgets:
                budget_count = int(budget * len(test_samples))
                intervention_samples = [sample_confidence_pair[0] for sample_confidence_pair in sample_confidence_pairs][:budget_count]
                intervention_samples_per_budget.append(intervention_samples)
        
        # Intervention by ranking with teacher confidence
        elif args.intervention_strategy == "correct_teacher_confidence":
            sample_confidence_pairs = []
            for i, test_sample in enumerate(test_samples):
                class_scores = tm.predict_single_confidence(test_sample, with_expl=True)
                if args.dataset == "strategyQA":
                    sample_confidence_pairs.append((i, class_scores[0])) if test_sample["answer"] == "yes" else sample_confidence_pairs.append((i, class_scores[1]))
                elif args.dataset == "ecqa":
                    sample_confidence_pairs.append((i, class_scores[int(test_sample["answer"]) - 1]))
            sample_confidence_pairs = sorted(sample_confidence_pairs, key=lambda x: x[1], reverse=True)
            for budget in budgets:
                budget_count = int(budget * len(test_samples))
                intervention_samples = [sample_confidence_pair[0] for sample_confidence_pair in sample_confidence_pairs][:budget_count]
                intervention_samples_per_budget.append(intervention_samples)
        
        # Intervention by Expected Utility with mental models (mm)
        elif args.intervention_strategy.startswith("mm"):
            _, _, no_inter_correct_scores, inter_correct_scores = mm.predict(test_samples)
            if args.intervention_strategy == "mm_no_inter":
                sample_no_inter_correct_scores = [(i, no_inter_correct_score) for i, no_inter_correct_score in enumerate(no_inter_correct_scores)]
                sample_no_inter_correct_scores = sorted(sample_no_inter_correct_scores, key=lambda x: x[1])
                for budget in budgets:
                    budget_count = int(budget * len(test_samples))
                    intervention_samples = [sample_no_inter_correct_score[0] for sample_no_inter_correct_score in sample_no_inter_correct_scores][:budget_count]
                    intervention_samples_per_budget.append(intervention_samples)

            elif args.intervention_strategy == "mm_inter":
                sample_inter_correct_scores = [(i, inter_correct_score) for i, inter_correct_score in enumerate(inter_correct_scores)]
                sample_inter_correct_scores = sorted(sample_inter_correct_scores, key=lambda x: x[1], reverse=True)
                for budget in budgets:
                    budget_count = int(budget * len(test_samples))
                    intervention_samples = [sample_inter_correct_score[0] for sample_inter_correct_score in sample_inter_correct_scores][:budget_count]
                    intervention_samples_per_budget.append(intervention_samples)

            elif args.intervention_strategy == "mm_both":
                sample_utility_correct_scores = [(i, (inter_correct_score - no_inter_correct_score)) for i, (no_inter_correct_score, inter_correct_score) in enumerate(zip(no_inter_correct_scores, inter_correct_scores))]
                sample_utility_correct_scores = sorted(sample_utility_correct_scores, key=lambda x: x[1], reverse=True) if not args.deceive else sorted(sample_utility_correct_scores, key=lambda x: x[1])
                for budget in budgets:
                    budget_count = int(budget * len(test_samples))
                    intervention_samples = [sample_utility_correct_score[0] for sample_utility_correct_score in sample_utility_correct_scores][:budget_count]
                    intervention_samples_per_budget.append(intervention_samples)
            else:
                assert False, "Intervention strategy not supported (inside mm)"
        else:
            assert False, "Intervention strategy not supported"
    else:
        intervention_samples_per_budget = [[]*len(budgets)]

    return intervention_samples_per_budget


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../datasets/strategyqa_dataset', type=str)
    parser.add_argument('--dataset', default='strategyQA', type=str)
    # parser.add_argument('--student_model_path', default='huggyllama/llama-7b', type=str)
    # parser.add_argument('--teacher_model_path', default='huggyllama/llama-65b', type=str)
    # parser.add_argument('--mental_model_path', default='huggyllama/llama-65b', type=str)
    parser.add_argument('--student_model_path', default='google/flan-t5-large', type=str)
    parser.add_argument('--teacher_model_path', default='google/flan-t5-xl', type=str)
    parser.add_argument('--mental_model_path', default='google/flan-t5-xl', type=str)

    parser.add_argument('--max_new_tokens_sm', default=100, type=int)
    parser.add_argument('--max_new_tokens_mm_no_inter', default=5, type=int)
    parser.add_argument('--max_new_tokens_mm_inter', default=5, type=int)
    parser.add_argument('--max_new_tokens_tm', default=100, type=int)
    parser.add_argument('--cache_dir', default='/data/home/swarnadeep/projects/MultiAgentReasoning/checkpoints', type=str)
    parser.add_argument('--num_beams', default=1, type=int)
    parser.add_argument('--ic_num', default=4, type=int)
    parser.add_argument('--use_explanations', default=True, type=bool)
    parser.add_argument('--intervention_strategy', default='mm_both', type=str)
    parser.add_argument('--intervention_action', default='teacher', type=str)
    parser.add_argument('--no_intervention_action', default='CoT', type=str)
    parser.add_argument('--teacher_expl_type', default='blind_teacher_CoT', type=str)
    parser.add_argument('--deceive', default=False, type=bool)

    parser.add_argument('--use_gold_label', default=True, type=bool)
    parser.add_argument('--results_file', default='../results/results_rq2.txt', type=str)

    args = parser.parse_args()

    if args.dataset == "strategyQA":
        dataset = StrategyQA(data_dir=args.data_dir)
    elif args.dataset == "ecqa":
        dataset = ECQA(data_dir=args.data_dir)
    elif args.dataset == "gsm8k":
        dataset = GSM8k(data_dir=args.data_dir)
    else:
        assert False, "Dataset not identified"

    test_samples = dataset.get_dev_samples() if args.dataset == "strategyQA" else dataset.get_test_samples()
    print(f"Number of test samples={len(test_samples)}")

    budgets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sm, tm, mm = None, None, None
    results_file = open(args.results_file, "w", encoding="utf-8-sig")
    for seed in [41, 42, 43]:
        random.seed(seed)
        train_samples = dataset.get_train_samples()
        if not sm:
            sm, tm, mm = load_all_models(args, train_samples)
        else:
            student_ics = random.sample(train_samples, args.ic_num)
            sm.set_ics(student_ics)

            if args.use_explanations:
                temp_tm = TeacherModel(model_name="human")
                if args.teacher_expl_type != "human_teacher":
                    teacher_ics = choose_in_context_samples_for_teacher_model(args, student_ics, train_samples, sm, temp_tm)
                    tm.set_ics(teacher_ics)

            if args.use_explanations and args.intervention_strategy.startswith("mm"):
                pre_ics, post_ics = choose_in_context_samples_for_mental_model(args, sm, tm, train_samples)
                mm.set_ics(pre_ics, post_ics)

        intervention_samples_per_budget = get_intervention_samples(args, mm, sm, tm, budgets, test_samples)

        questions, labels, predictions_per_budget, explanations_per_budget = sm.predict(test_samples, intervention_samples_per_budget, tm)
        if not args.use_explanations:
            accuracy = compute_accuracy(labels, predictions_per_budget[0])
            print(f"Accuracy = {accuracy}")
            results_file.write(f"Seed = {seed}\n")
            results_file.write(f"Accuracy = {accuracy}\n")
            results_file.flush()
            os.fsync(results_file.fileno())
        else:
            for budget_index, budget in enumerate(budgets):
                predictions, explanations = predictions_per_budget[budget_index], explanations_per_budget[budget_index]
                accuracy = compute_accuracy(labels, predictions)
                print(f"Accuracy for budget {budget} = {accuracy}")
                results_file.write(f"Seed = {seed}\n")
                results_file.write(f"Accuracy for budget {budget} = {accuracy}\n")
                results_file.flush()
                os.fsync(results_file.fileno())
