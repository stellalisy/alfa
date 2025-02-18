import os
import subprocess
import argparse

os.environ["IS_ALPACA_EVAL_2"] = "True"

EVAL_LIST = ["ifeval", "truthfulqa", "gsm", "bbh", "math", "alpaca"]
RESULTS_DIR = "/mmfs1/gscratch/stf/stelli/health-q/results"
DATA_DIR = "/gscratch/argon/stelli/health-q/data"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--results_base_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--results_name", type=str, default="baselines/llama-3.1-8b-instruct")
    parser.add_argument("--eval_list", nargs="+", choices=EVAL_LIST, default=EVAL_LIST, help="List of evaluations to run")
    parser.add_argument("--dry_run", action="store_true", help="Just print the eval commands")
    parser.add_argument("--no_cot", action="store_true", help="Don't use cot")
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip if results already exist")
    return parser.parse_args()

def main(args):
    print(f"Running evaluations for {args.model_name} on {args.eval_list}")
    print(f"Results will be saved in {os.path.join(args.results_base_dir, '<benchmark_name>', args.results_name)}")

    def should_skip(save_dir):
        return args.skip_if_exists and os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0

    for eval_name in args.eval_list:
        if eval_name == 'ifeval':
            save_dir = os.path.join(args.results_base_dir, args.results_name, 'ifeval')
            if should_skip(save_dir):
                print(f"Skipping IF-Eval as results already exist in {save_dir}")
                continue
            cmd = [
                "python", "-m", "eval.ifeval.run_eval",
                "--data_dir", f"{DATA_DIR}/eval/ifeval/",
                "--save_dir", save_dir,
                "--model_name_or_path", args.model_name,
                "--tokenizer_name_or_path", args.model_name,
                "--use_chat_format",
                "--chat_formatting_function", "eval.templates.create_prompt_with_huggingface_tokenizer_template",
                "--use_vllm"
            ]
            print("Evaluating", args.model_name, "on IF-Eval")
            if args.dry_run:
                print(" ".join(cmd))
            else:
                subprocess.run(cmd)

        elif eval_name == 'truthfulqa':
            save_dir = os.path.join(args.results_base_dir, args.results_name, 'truthfulqa')
            if should_skip(save_dir):
                print(f"Skipping TruthfulQA as results already exist in {save_dir}")
                continue
            cmd = [
                "python", "-m", "eval.truthfulqa.run_eval",
                "--data_dir", f"{DATA_DIR}/eval/truthfulqa",
                "--save_dir", save_dir,
                "--model_name_or_path", args.model_name,
                "--tokenizer_name_or_path", args.model_name,
                "--metrics", "truth", "info", "mc",
                "--preset", "qa",
                "--hf_truth_model_name_or_path", "allenai/truthfulqa-truth-judge-llama2-7B",
                "--hf_info_model_name_or_path", "allenai/truthfulqa-info-judge-llama2-7B",
                "--use_chat_format",
                "--chat_formatting_function", "eval.templates.create_prompt_with_huggingface_tokenizer_template",
                "--eval_batch_size", "64"
            ]
            print("Evaluating", args.model_name, "on TruthfulQA")
            if args.dry_run:
                print(" ".join(cmd))
            else:
                subprocess.run(cmd)

        elif eval_name == 'gsm':
            cot_suffix = '-cot-8shot' if not args.no_cot else ''
            save_dir = os.path.join(args.results_base_dir, args.results_name, f'gsm{cot_suffix}')
            if should_skip(save_dir):
                print(f"Skipping GSM8K as results already exist in {save_dir}")
                continue
            cmd = [
                "python", "-m", "eval.gsm.run_eval",
                "--data_dir", f"{DATA_DIR}/eval/gsm/",
                "--max_num_examples", "200",
                "--save_dir", save_dir,
                "--model_name_or_path", args.model_name,
                "--tokenizer_name_or_path", args.model_name,
                "--use_vllm"
            ]
            if not args.no_cot:
                cmd += ["--n_shot", "8"]

            print("Evaluating", args.model_name, "on GSM8K")
            if args.dry_run:
                print(" ".join(cmd))
            else:
                subprocess.run(cmd)

        elif eval_name == 'bbh':
            cot_suffix = '-cot' if not args.no_cot else ''
            save_dir = os.path.join(args.results_base_dir, args.results_name, f"bbh{cot_suffix}")
            if should_skip(save_dir):
                print(f"Skipping BBH as results already exist in {save_dir}")
                continue
            cmd = [
                "python", "-m", "eval.bbh.run_eval",
                "--data_dir", f"{DATA_DIR}/eval/bbh",
                "--save_dir", save_dir,
                "--model_name_or_path", args.model_name,
                "--tokenizer_name_or_path", args.model_name,
                "--max_num_examples_per_task", "40",
                "--use_chat_format",
                "--chat_formatting_function", "eval.templates.create_prompt_with_huggingface_tokenizer_template",
                "--use_vllm"
            ]
            if args.no_cot:
                cmd += ["--no_cot"]
            print("Evaluating", args.model_name, "on BBH")
            if args.dry_run:
                print(" ".join(cmd))
            else:
                subprocess.run(cmd)

        elif eval_name == 'math':
            cot_suffix = '-cot-4shot' if not args.no_cot else ''
            save_dir = os.path.join(args.results_base_dir, args.results_name, f"math{cot_suffix}")
            if should_skip(save_dir):
                print(f"Skipping MATH as results already exist in {save_dir}")
                continue
            cmd = [
                "python", "-m", "eval.MATH.run_eval",
                "--data_dir", f"{DATA_DIR}/eval/MATH",
                "--max_num_examples", "200",
                "--save_dir", save_dir,
                "--model_name_or_path", args.model_name,
                "--tokenizer_name_or_path", args.model_name,
                "--use_chat_format",
                "--chat_formatting_function", "eval.templates.create_prompt_with_huggingface_tokenizer_template",
                "--use_vllm"
            ]
            if not args.no_cot:
                cmd += ["--n_shot", "4"]
            print("Evaluating", args.model_name, "on MATH")
            if args.dry_run:
                print(" ".join(cmd))
            else:
                subprocess.run(cmd)

        elif eval_name == 'alpaca':
            from openai import OpenAI
            vllm_port = os.environ.get("VLLM_SERVER_PORT", 8000)
            client = OpenAI(
                base_url=f"http://localhost:{vllm_port}/v1",
                api_key="token-abc123",
            )
            annotator = "alpaca_eval_vllm_llama3.1_70b_fn"
            try:
                models = client.models.list()
                model_names = [x.id for x in models.data]
                if "meta-llama/Llama-3.1-70B-Instruct" in model_names:
                    annotator = "alpaca_eval_vllm_server_llama3.1_70b_fn"
            except:
                pass
                
            save_dir = os.path.join(args.results_base_dir, args.results_name, 'alpaca_farm_fn-llama3.1-70b')
    
            if should_skip(save_dir):
                print(f"Skipping AlpacaEval as results already exist in {save_dir}")
                continue
            cmd = [
                "python", "-m", "eval.alpaca_farm.run_eval",
                "--save_dir", save_dir,
                "--model_name_or_path", args.model_name,
                "--tokenizer_name_or_path", args.model_name,
                "--use_vllm",
                "--use_chat_format",
                "--chat_formatting_function", "eval.templates.create_prompt_with_huggingface_tokenizer_template",
                "--annotators_config", annotator
            ]
            print("Evaluating", args.model_name, "on AlpacaEval 2")
            if args.dry_run:
                print(" ".join(cmd))
            else:
                subprocess.run(cmd)


if __name__ == "__main__":
    args = parse_args()
    main(args)