import argparse
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import random
import os
from copy import deepcopy


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
                _, teacher_explanation = tm.predict_single(test_sample=train_sample)
                student_prediction, _ = sm.predict_single(test_sample=train_sample, tm=tm, inside_mm=False, intervene=True)
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
        print("Done")

        return pre_in_context_samples, post_in_context_samples


def choose_in_context_samples_for_teacher_model(args, student_ics, train_samples, sm, tm):
    teacher_ics = []
    if args.teacher_expl_type == "blind_teacher_CoT" or args.teacher_expl_type == "blind_teacher_rationalize":
        teacher_ics = student_ics
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
        assert False, "ToM type not recognized"

    return teacher_ics

def load_all_models(args, train_samples):
    student_in_context_samples = random.sample(train_samples, args.ic_num)
    print("Loading student model!!!")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model_path, cache_dir=args.cache_dir, use_fast=False)
    smodel = AutoModelForCausalLM.from_pretrained(args.student_model_path, cache_dir=args.cache_dir, device_map="auto", torch_dtype=torch.float16)
    print("Done")

    sm = StudentModel(model_name=args.student_model_path,
                      model=smodel,
                      dataset=args.dataset,
                      tokenizer=tokenizer,
                      in_context_samples=student_in_context_samples,
                      use_explanations=args.use_explanations,
                      intervention_strategy=args.intervention_strategy,
                      intervention_action=args.intervention_action,
                      no_intervention_action=args.no_intervention_action,
                      max_new_tokens=args.max_new_tokens_sm
                      )

    print("Loading teacher model!!!")
    if args.use_explanations:
        tm = TeacherModel(model_name="human")
        tmodel = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, cache_dir=args.cache_dir, device_map="auto", torch_dtype=torch.float16)

        if args.teacher_expl_type != "human_teacher":
            teacher_ics = choose_in_context_samples_for_teacher_model(args, student_in_context_samples, train_samples, sm, tm)

            tm = TeacherModel(model_name=args.teacher_model_path,
                              model=tmodel,
                              dataset=args.dataset,
                              expl_type=args.teacher_expl_type,
                              tokenizer=tokenizer,
                              in_context_samples=teacher_ics,
                              max_new_tokens=args.max_new_tokens_tm)            

    else:
        return sm, None, None

    if args.use_explanations and args.intervention_strategy.startswith("mm"):
        print("Loading mental model!!!")
        pre_ics, post_ics = choose_in_context_samples_for_mental_model(args, sm, tm, train_samples)
        mm = MentalModel(model_name=args.mental_model_path,
                         model=tmodel,
                         dataset=args.dataset,
                         tokenizer=tokenizer,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../datasets/strategyqa_dataset', type=str)
    parser.add_argument('--dataset', default='strategyQA', type=str)
    parser.add_argument('--student_model_path', default='huggyllama/llama-7b', type=str)
    parser.add_argument('--teacher_model_path', default='huggyllama/llama-65b', type=str)
    parser.add_argument('--mental_model_path', default='huggyllama/llama-65b', type=str)

    parser.add_argument('--max_new_tokens_sm', default=100, type=int)
    parser.add_argument('--max_new_tokens_mm_no_inter', default=5, type=int)
    parser.add_argument('--max_new_tokens_mm_inter', default=5, type=int)
    parser.add_argument('--max_new_tokens_tm', default=100, type=int)
    parser.add_argument('--cache_dir', default='/data/home/swarnadeep/projects/MultiAgentReasoning/checkpoints', type=str)
    parser.add_argument('--num_beams', default=1, type=int)
    parser.add_argument('--ic_num', default=2, type=int)
    parser.add_argument('--use_explanations', default=True, type=bool)

    parser.add_argument('--intervention_strategy', default='mm_both', type=str)
    parser.add_argument('--intervention_action', default='teacher', type=str)
    parser.add_argument('--no_intervention_action', default='CoT', type=str)
    parser.add_argument('--teacher_expl_type', default='useful_teacher', type=str)
    parser.add_argument('--use_gold_label', default=True, type=bool)

    parser.add_argument('--output_dir', default='../all_outputs/', type=str)
    parser.add_argument('--results_file', default='../results_RQ4.txt', type=str)

    parser.add_argument('--explained_samples', default="teacher", type=str)

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
    train_samples = dataset.get_train_samples()

    budgets = [2, 4, 6, 8, 10]
    sm, tm, mm = None, None, None
    results_file = open(args.results_file, "w", encoding="utf-8-sig")
    for seed in [41, 42, 43]:
        random.seed(seed)

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

        for budget in budgets:
            new_student_ics = []
            intervention_samples = random.sample(train_samples, budget)
            for intervention_sample in intervention_samples:
                teacher_prediction, teacher_explanation = tm.predict_single(test_sample=intervention_sample)
                additional_student_ic = deepcopy(intervention_sample)

                if args.explained_samples == "teacher":
                    additional_student_ic["gold_explanation"] = teacher_explanation
                    additional_student_ic["answer"] = teacher_prediction

                new_student_ics.append(additional_student_ic)
                
            sm.in_context_samples = new_student_ics

            student_predictions_no_intervention, labels = [], []
            for test_sample in test_samples:
                labels.append(test_sample["answer"])
                student_prediction, student_explanation = sm.predict_single(test_sample=test_sample, tm=tm, inside_mm=False, intervene=False)
                student_predictions_no_intervention.append(student_prediction)

            accuracy = compute_accuracy(labels, student_predictions_no_intervention)
            print(f"Accuracy for budget {budget} = {accuracy}")

            results_file.write(f"Seed = {seed}\n")
            results_file.write(f"Accuracy for budget {budget} = {accuracy}\n")

            results_file.flush()
            os.fsync(results_file.fileno())

