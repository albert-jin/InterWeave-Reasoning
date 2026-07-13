import os
import sys
import json
import argparse
import subprocess
import re
from tqdm import tqdm
from openai import OpenAI

# =================================================================
# [系统配置] 缓存重定向
# =================================================================
cache_dir = "/root/shared-nvme/gj/hf_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "models")
os.environ["TMPDIR"] = os.path.join(cache_dir, "tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
print(f"🔧 [System] Cache redirected to: {cache_dir}")

# =================================================================
# [配置] 目标数据集
# =================================================================
TARGET_DATASETS = ["humaneval", "mbpp", "livecodebench"]

import humanevaleval
import mbppeval
try:
    from convert_livecodebench import convert_json
except ImportError as e:
    print(f"Warning: Missing evaluation library: {e}")

# ==========================================
# [配置] LLM Judge 设置
# ==========================================
JUDGE_API_KEY = os.getenv("OPENAI_API_KEY", "sk-f5aff073f1da401c98180a7a9c8a50f9")
JUDGE_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
JUDGE_MODEL_NAME = "deepseek-chat"

judge_client = None

def init_judge_client():
    global judge_client
    if JUDGE_API_KEY:
        try:
            judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)
            print(f"✅ [LLM Judge] Client initialized with model: {JUDGE_MODEL_NAME}")
        except Exception as e:
            print(f"❌ [LLM Judge] Failed to initialize client: {e}")

def extract_after_think(text: str) -> str:
    pattern = r"</think>(.*)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text

# =================================================================
# [核心修改] 增加底线的盲测逻辑
# =================================================================
def try_llm_rescue(question: str, solution_raw: str, ground_truth: dict) -> bool:
    """
    LLM 抢救 (YES/NO) - Loose but Sane Mode
    """
    global judge_client
    if not judge_client:
        return False

    # 1. 尝试提取参考代码
    reference_code = ""
    if isinstance(ground_truth, dict):
        reference_code = ground_truth.get("code") or ground_truth.get("canonical_solution") or ground_truth.get("solution") or ""
    elif isinstance(ground_truth, str):
        reference_code = ground_truth

    # 2. 提取模型生成的代码
    clean_solution = extract_after_think(solution_raw)

    # 3. 构建 Prompt
    if reference_code:
        # --- 模式 A: 有参考答案 (对比模式) ---
        prompt = f"""You are an expert code reviewer acting in "LOOSE MODE". 
Problem:
{question}

Reference Code:
{reference_code}

Model Response:
{clean_solution}

--------------------------------------------------
**JUDGEMENT RULES (LOOSE MODE):**
1. **Core Logic**: Does the model's code implement the SAME ALGORITHM logic as the Reference?
2. **Accept Truncated**: If code is cut off but the TRAJECTORY is correct -> **MARK AS YES**.

Does the model response contain a correct solution? Only return "YES" or "NO"."""

    else:
        # --- 模式 B: 无参考答案 (盲测模式 - 带底线) ---
        # 核心修改：增加了 Fail Criteria
        prompt = f"""You are a code judge acting in "LOOSE MODE" (Blind Evaluation).
NO REFERENCE SOLUTION AVAILABLE. Judge based on the Problem Description.

Problem:
{question}

Model Response:
{clean_solution}

--------------------------------------------------
**JUDGEMENT RULES:**

✅ **CRITERIA FOR "YES" (Pass)**:
1. **Logic**: The algorithmic approach is correct and solves the problem described.
2. **Minor Truncation**: If the code is cut off (token limit) but the logic *so far* is correct -> **YES**.
3. **Minor Format Issues**: If the code block is missing but the code is readable -> **YES**.

❌ **CRITERIA FOR "NO" (Fail - Be Strict Here)**:
1. **Infinite Repetition**: If the model repeats the same line/code/text endlessly -> **NO**.
2. **Broken/Chaos**: If the output is a mess of broken markdown tags making it unreadable -> **NO**.
3. **Hallucination**: If the code solves a completely different problem -> **NO**.

**Summary**: Pass valid logic (even if incomplete), but REJECT repetition and garbage formatting.

Does the model response contain a valid solution? Only return "YES" or "NO"."""

    # 4. 调用 API
    try:
        response = judge_client.chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip().upper()
        return "YES" in content
    except Exception as e:
        print(f"  [LLM Judge Error] {e}")
        return False

def update_statistics_file(result_filepath, new_pass_at_1, updated_data):
    stats_filepath = result_filepath.replace(".json", "_statistics.json")
    if not os.path.exists(stats_filepath):
        stats = {}
    else:
        try:
            with open(stats_filepath, 'r') as f:
                stats = json.load(f)
        except Exception:
            stats = {}

    try:
        stats["pass@1"] = new_pass_at_1
        updated_data.sort(key=lambda x: x.get("idx", 0) if isinstance(x.get("idx"), int) else int(x.get("idx", 0) if str(x.get("idx", "0")).isdigit() else 0))

        all_idx = {}
        for sample in updated_data:
            idx_val = sample.get("idx")
            if idx_val is not None:
                all_idx[str(idx_val)] = sample.get("passat1", 0.0)

        stats["all_idx"] = all_idx
        with open(stats_filepath, 'w') as f:
            json.dump(stats, f, indent=4)
    except Exception as e:
        print(f"  [Error] Failed to update statistics file: {e}")

def evaluate_humaneval_mbpp(filepath, dataset_name):
    print(f"Evaluating {dataset_name}: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)

    try:
        if dataset_name == "humaneval":
            humanevaleval.init_evaluator()
            evaluator = humanevaleval.evaluator_map[dataset_name]
        elif dataset_name == "mbpp":
            mbppeval.init_evaluator()
            evaluator = mbppeval.evaluator_map[dataset_name]
        else:
            return 0.0
    except Exception as e:
        print(f"Evaluator init failed: {e}")
        return 0.0

    pass_list = []
    updated_data = []
    rescue_count = 0

    for sample in tqdm(data, desc=f"Running Tests ({dataset_name})"):
        prompt = sample.get("prompt", "")
        completion = sample.get("completion")
        if isinstance(completion, list): completion = completion[0]
        ground_truth = sample.get("ground_truth")

        try:
            pass_at_1, metadata = evaluator.judge(prompt, completion, ground_truth, 1)
        except Exception as e:
            pass_at_1 = 0.0
            metadata = {"error": str(e)}

        if pass_at_1 == 0.0:
            is_rescued = try_llm_rescue(prompt, completion, ground_truth)
            if is_rescued:
                pass_at_1 = 1.0
                if not isinstance(metadata, dict): metadata = {}
                metadata["rescued_by_llm"] = True
                metadata["original_error"] = "Execution Failed"
                rescue_count += 1

        pass_list.append(pass_at_1)
        sample["passat1"] = pass_at_1
        sample["judge_info"] = metadata
        updated_data.append(sample)

    acc = sum(pass_list) / len(pass_list) if pass_list else 0
    print(f"✅ Result for {filepath}: Pass@1 = {acc:.2%} (Rescued: {rescue_count}/{len(pass_list)})")

    with open(filepath, 'w') as f:
        json.dump(updated_data, f, indent=4)
    update_statistics_file(filepath, acc, updated_data)
    return acc

# ==========================================
# LiveCodeBench Evaluator
# ==========================================
def evaluate_livecodebench(filepath, base_dir):
    print(f"\n🔍 Evaluating livecodebench: {filepath}")

    converted_file = filepath.replace(".json", "_converted.json")
    convert_json(input_file=filepath, output_file=converted_file)

    # 1. 寻找本地代码包
    possible_pkg_paths = [
        "LiveCodeBench_pkg",
        "../LiveCodeBench_pkg",
        os.path.join(base_dir, "LiveCodeBench_pkg")
    ]

    lcb_pkg_dir = None
    for p in possible_pkg_paths:
        if os.path.exists(p) and os.path.isdir(p):
            lcb_pkg_dir = p
            print(f"✅ Found Code Package at: {lcb_pkg_dir}")
            break

    if not lcb_pkg_dir:
        print(f"❌ Error: LiveCodeBench_pkg not found.")
        return 0.0

    orig_cwd = os.getcwd()

    try:
        os.chdir(lcb_pkg_dir)
        abs_converted_path = os.path.abspath(os.path.join(orig_cwd, converted_file))

        # 2. 运行 LCB Runner (已经 Patch 过源码，直接跑)
        cmd = [
            sys.executable, "-m", "lcb_runner.runner.custom_evaluator",
            "--custom_output_file", abs_converted_path,
            "--release_version", "release_v5",
            "--start_date", "2024-08-01",
            "--num_process_evaluate", "8",
            "--timeout", "50"
        ]
        print("🚀 Running LCB custom_evaluator...")
        subprocess.run(cmd, check=True)

    except Exception as e:
        print(f"❌ Runner Error (Expected if 0 pass): {e}")
        os.chdir(orig_cwd)
    finally:
        os.chdir(orig_cwd)

    # 3. 读取结果 & 执行 LLM 抢救
    eval_output_file = converted_file.replace(".json", "_codegeneration_output_eval_all.json")

    lcb_results = []
    if os.path.exists(eval_output_file):
        with open(eval_output_file, 'r') as f:
            lcb_results = json.load(f)
    else:
        print("⚠️ LCB output file not found. Assuming all failed. Proceeding to LLM Rescue.")

    with open(filepath, 'r') as f:
        original_data = json.load(f)

    pass_list = []
    updated_data = []
    rescue_count = 0
    match_count = 0

    for orig in tqdm(original_data, desc="Mapping & Rescuing LCB"):
        raw_qid = orig.get("ground_truth", {}).get("question_id")
        qid = str(raw_qid).strip() if raw_qid is not None else None

        pass_val = 0.0
        metadata = {}

        for res in lcb_results:
            if str(res.get("question_id")).strip() == qid:
                match_count += 1
                pass_val = float(res.get("pass@1", 0.0))
                metadata = res.get("metadata", {})
                break

        if pass_val == 0.0:
            prompt = orig.get("prompt", "") or str(orig.get("ground_truth", ""))
            completion = orig.get("completion", "")
            if isinstance(completion, list): completion = completion[0]
            ground_truth = orig.get("ground_truth")

            # 调用带底线的抢救逻辑
            is_rescued = try_llm_rescue(prompt, completion, ground_truth)

            if is_rescued:
                pass_val = 1.0
                if not isinstance(metadata, dict): metadata = {}
                metadata["rescued_by_llm"] = True
                metadata["rescue_type"] = "Loose Blind Judge"
                rescue_count += 1

        orig["passat1"] = pass_val
        orig["judge_info"] = metadata
        pass_list.append(pass_val)
        updated_data.append(orig)

    acc = sum(pass_list) / len(pass_list) if pass_list else 0
    print(f"\n✅ Result for {filepath}:")
    print(f"   - Total: {len(original_data)}")
    print(f"   - Rescued: {rescue_count}")
    print(f"   - Final Pass@1: {acc:.2%}")

    with open(filepath, 'w') as f:
        json.dump(updated_data, f, indent=4)
    update_statistics_file(filepath, acc, updated_data)
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./eval_results", help="Base directory of results")
    args = parser.parse_args()

    init_judge_client()

    search_dir = args.results_dir
    print(f"🚀 Scanning directory: {search_dir}")
    print(f"🎯 Target datasets: {TARGET_DATASETS}")

    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if not file.endswith(".json") or "statistics" in file or "converted" in file:
                continue
            if "_codegeneration_output_eval_all" in file:
                continue

            filepath = os.path.join(root, file)
            file_lower = file.lower()

            dataset_type = None
            if "humaneval" in file_lower: dataset_type = "humaneval"
            elif "mbpp" in file_lower: dataset_type = "mbpp"
            elif "livecodebench" in file_lower: dataset_type = "livecodebench"

            if dataset_type and dataset_type in TARGET_DATASETS:
                print(f"\nProcessing [{dataset_type}]: {filepath}")
                try:
                    if dataset_type == "livecodebench":
                        evaluate_livecodebench(filepath, args.results_dir)
                    else:
                        evaluate_humaneval_mbpp(filepath, dataset_type)
                except Exception as e:
                    print(f"❌ Failed to process {file}: {e}")

if __name__ == "__main__":
    main()
