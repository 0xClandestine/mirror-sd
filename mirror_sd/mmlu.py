"""MMLU benchmark for speculative decoding.

Evaluates accuracy and throughput on the MMLU (Massive Multitask Language
Understanding) benchmark using 5-shot prompting.

Usage:
    python -m mirror_sd.mmlu --model Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16
    python -m mirror_sd.mmlu --model Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 --subjects abstract_algebra,anatomy
    python -m mirror_sd.mmlu --model Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 --baseline-only
"""

import argparse
import time

import mlx.core as mx
from datasets import load_dataset
from mlx_lm import load as mlx_load
from mlx_lm.models import cache as cache_module

from .generate import spec_generate
from .loader import load_dflash_model
from .prompt import format_prompt, get_stop_token_ids


MMLU_CHOICES = ["A", "B", "C", "D"]
GEN_TOKENS = 8

FEW_SHOT_TEMPLATE = """The following are multiple choice questions (with answers) about {subject}.

{few_shot}Question: {question}
A. {a}
B. {b}
C. {c}
D. {d}
Answer:"""

SHOT_TEMPLATE = """Question: {question}
A. {a}
B. {b}
C. {c}
D. {d}
Answer: {answer}

"""


def format_mmlu_question(question: str, choices: list, subject: str,
                         dev_set: list, n_shots: int = 5) -> str:
    shots = ""
    for ex in dev_set[:n_shots]:
        answer_letter = MMLU_CHOICES[ex["answer"]]
        shots += SHOT_TEMPLATE.format(
            question=ex["question"],
            a=ex["choices"][0], b=ex["choices"][1],
            c=ex["choices"][2], d=ex["choices"][3],
            answer=answer_letter,
        )
    return FEW_SHOT_TEMPLATE.format(
        subject=subject.replace("_", " "),
        few_shot=shots,
        question=question,
        a=choices[0], b=choices[1], c=choices[2], d=choices[3],
    )


def baseline_mmlu(model, tokenizer, prompt: str, use_chat: bool, enable_thinking: bool = False):
    formatted = format_prompt(tokenizer, prompt, enable_thinking=enable_thinking) if use_chat else prompt
    tokens = tokenizer.encode(formatted)
    input_ids = mx.array(tokens)[None]
    ctx_len = input_ids.shape[1]
    cache = cache_module.make_prompt_cache(model)

    t0 = time.perf_counter()
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    mx.eval([c.state for c in cache])
    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
    mx.eval(next_token)
    generated = [int(next_token[0, 0])]

    for _ in range(GEN_TOKENS - 1):
        logits = model(mx.array([[generated[-1]]]), cache=cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(next_token)
        generated.append(int(next_token[0, 0]))
    t1 = time.perf_counter()

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    elapsed_ms = (t1 - t0) * 1000
    return text, ctx_len, elapsed_ms


def spec_mmlu(model, draft_model, tokenizer, prompt: str,
              use_chat: bool, eos_ids, **spec_kwargs):
    formatted = format_prompt(tokenizer, prompt, enable_thinking=spec_kwargs.get("enable_thinking", False)) if use_chat else prompt
    tokens = tokenizer.encode(formatted)
    input_ids = mx.array(tokens)[None]
    ctx_len = input_ids.shape[1]

    t0 = time.perf_counter()
    output_ids, stats, _, _, _ = spec_generate(
        model, draft_model, input_ids,
        max_new_tokens=GEN_TOKENS,
        temperature=0.0,
        stop_token_ids=eos_ids,
        adaptive_block=spec_kwargs.get("adaptive_block", True),
        kod=spec_kwargs.get("kod", False),
    )
    t1 = time.perf_counter()

    gen_only = output_ids[0, input_ids.shape[1]:].tolist() if output_ids.ndim == 2 else output_ids.tolist()
    answer = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    elapsed_ms = (t1 - t0) * 1000
    return answer, ctx_len, elapsed_ms, stats


def run_mmlu(args):
    print(f"Loading target: {args.model}")
    model, tokenizer = mlx_load(args.model)

    draft_model = None
    config = None
    if not args.baseline_only:
        print(f"Loading draft:  {args.draft}")
        draft_model, config = load_dflash_model(args.draft, quantize=args.quantize_draft)
        if args.block_size is not None:
            config.block_size = args.block_size
            draft_model.block_size = args.block_size

    use_chat = not args.raw_prompt
    eos_ids = get_stop_token_ids(tokenizer) or None

    print("Loading MMLU dataset...")
    test_ds = load_dataset("cais/mmlu", "all", split="test")
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")

    all_subjects = sorted(set(test_ds["subject"]))
    if args.subjects:
        selected = [s.strip() for s in args.subjects.split(",")]
        subjects = [s for s in selected if s in all_subjects]
        unknown = [s for s in selected if s not in all_subjects]
        if unknown:
            print(f"Warning: unknown subjects skipped: {unknown}")
    else:
        subjects = all_subjects

    dev_by_subject = {}
    for ex in dev_ds:
        s = ex["subject"]
        if s not in dev_by_subject:
            dev_by_subject[s] = []
        dev_by_subject[s].append(ex)

    n_shots = args.n_shots
    mode = "BASELINE" if args.baseline_only else "DFLASH" + ("+KOD" if args.kod else "+ADAPTIVE")

    print(f"\n{'='*70}")
    print(f"  MMLU Benchmark ({mode})")
    print(f"  Subjects: {len(subjects)}, Shots: {n_shots}")
    print(f"{'='*70}")
    print(f"  {'Subject':40s} {'N':>4s} {'Acc':>6s} {'Ctx':>5s} {'Q/ms':>7s} {'tok/s':>7s}")
    print(f"  {'-'*70}")

    subject_results = {}
    total_correct = 0
    total_count = 0
    total_ms = 0.0
    total_gen_tokens = 0
    t0_all = time.perf_counter()

    for subj in subjects:
        examples = [ex for ex in test_ds if ex["subject"] == subj]
        dev_examples = dev_by_subject.get(subj, [])
        correct = 0
        count = 0
        subj_ms = 0.0
        subj_ctx_len = 0

        for ex in examples:
            prompt = format_mmlu_question(
                ex["question"], ex["choices"], subj, dev_examples, n_shots
            )

            if args.baseline_only:
                answer, ctx_len, elapsed_ms = baseline_mmlu(model, tokenizer, prompt, use_chat, args.think)
            else:
                answer, ctx_len, elapsed_ms, stats = spec_mmlu(
                    model, draft_model, tokenizer, prompt, use_chat, eos_ids,
                    adaptive_block=not args.no_adaptive,
                    kod=args.kod,
                    enable_thinking=args.think,
                )

            predicted = answer[0].upper() if answer else "?"
            target = MMLU_CHOICES[ex["answer"]]
            is_correct = predicted == target

            correct += int(is_correct)
            count += 1
            subj_ms += elapsed_ms
            subj_ctx_len += ctx_len
            total_gen_tokens += 1

            if count % 50 == 0:
                acc = 100.0 * correct / count
                avg_ctx = subj_ctx_len / count
                q_ms = subj_ms / count
                print(f"  {subj:40s} {count:4d}  {acc:5.1f}% {avg_ctx:5.0f} {q_ms:7.1f}", flush=True)

        acc = 100.0 * correct / max(count, 1)
        avg_ctx = subj_ctx_len / max(count, 1)
        q_ms = subj_ms / max(count, 1)
        decode_tps = 1000.0 / q_ms if q_ms > 0 else 0

        subject_results[subj] = (correct, count, acc, q_ms, avg_ctx)
        total_correct += correct
        total_count += count
        total_ms += subj_ms
        print(f"  {subj:40s} {count:4d}  {acc:5.1f}% {avg_ctx:5.0f} {q_ms:7.1f} {decode_tps:7.1f}")

    elapsed = time.perf_counter() - t0_all
    overall_acc = 100.0 * total_correct / max(total_count, 1)
    overall_q_ms = total_ms / max(total_count, 1)
    overall_tps = 1000.0 / overall_q_ms if overall_q_ms > 0 else 0
    avg_ctx_all = sum(subject_results[s][4] * subject_results[s][1] for s in subjects) / max(total_count, 1)

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Mode:           {mode}")
    print(f"  Subjects:       {len(subjects)}")
    print(f"  Questions:       {total_count}")
    print(f"  Accuracy:       {overall_acc:.1f}%")
    print(f"  Avg ctx len:    {avg_ctx_all:.0f} tokens")
    print(f"  Avg time/q:     {overall_q_ms:.1f} ms")
    print(f"  Throughput:      {overall_tps:.1f} q/s")
    print(f"  Wall time:      {elapsed:.1f}s")
    print(f"  Chat fmt:       {'off (raw)' if args.raw_prompt else ('thinking' if args.think else '/no_think')}")

    if args.subjects is None and len(subjects) > 1:
        print(f"\n  Category accuracy breakdown:")
        stem_cats = {
            "STEM": ["abstract_algebra", "anatomy", "astronomy", "college_biology",
                      "college_chemistry", "college_computer_science", "college_mathematics",
                      "college_physics", "computer_security", "conceptual_physics",
                      "electrical_engineering", "elementary_mathematics", "high_school_biology",
                      "high_school_chemistry", "high_school_computer_science",
                      "high_school_mathematics", "high_school_physics",
                      "high_school_statistics", "machine_learning"],
            "Humanities": ["formal_logic", "high_school_european_history", "high_school_us_history",
                           "high_school_world_history", "history", "international_law",
                           "jurisprudence", "logical_fallacies", "moral_disputes",
                           "moral_scenarios", "philosophy", "prehistory",
                           "professional_law", "world_religions"],
            "Social Science": ["business_ethics", "econometrics", "high_school_government_and_politics",
                               "high_school_macroeconomics", "high_school_microeconomics",
                               "human_sexuality", "macroeconomics", "microeconomics",
                               "professional_accounting", "public_relations", "security_studies",
                               "sociology", "us_foreign_policy"],
            "Other": ["clinical_knowledge", "college_medicine", "global_facts", "management",
                      "marketing", "medical_genetics", "nutrition", "professional_medicine",
                      "virology"],
        }
        for cat_name, cat_subjects in stem_cats.items():
            cat_correct = sum(subject_results.get(s, (0,1,0,0,0))[0] for s in cat_subjects if s in subject_results)
            cat_total = sum(subject_results.get(s, (0,1,0,0,0))[1] for s in cat_subjects if s in subject_results)
            if cat_total > 0:
                cat_acc = 100.0 * cat_correct / cat_total
                print(f"    {cat_name:20s} {cat_acc:5.1f}% ({cat_correct}/{cat_total})")


def main():
    parser = argparse.ArgumentParser(description="MMLU benchmark for speculative decoding")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--draft", type=str, default=None)
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject list (default: all)")
    parser.add_argument("--n-shots", type=int, default=5, help="Number of few-shot examples (default: 5)")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--baseline-only", action="store_true", help="Run baseline only (no speculative decoding)")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive block size")
    parser.add_argument("--kod", action="store_true", help="Kelly-Optimal Drafting")
    parser.add_argument("--quantize-draft", type=int, default=None, choices=[4, 8])
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode")
    args = parser.parse_args()

    if not args.baseline_only and args.draft is None:
        parser.error("--draft is required unless --baseline-only is set")

    run_mmlu(args)


if __name__ == "__main__":
    main()
