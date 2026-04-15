import os
import time
import re
import openai

def extract_problem(text: str) -> str | None:
    start_tag = "<problem>"
    end_tag = "</problem>"

    if start_tag in text and end_tag in text:
        return text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
    return text

GENERAL_ORM_PROMPT = """You are an expert in verifying if two answers are the same.
Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are equivalent.
Your task is to determine if two answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.

Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
2) Then provide your final answer in the form of: [[YES]] or [[NO]]
"""

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""


def call_oai_rm_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    model_id: str = "gpt-4o",
    retry_count: int = 1000000000,
):
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{os.getenv('VERIFIER_HOST')}:{os.getenv('VERIFIER_PORT')}/v1"
    client = openai.OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    backoff = 1
    retry_count = int(retry_count)

    for _ in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                n=n,
            )
            break
        except Exception as exc:
            if "429" in str(exc):
                print("Retry due to rate limit: ", exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)
                continue
            print("Exception: ", exc)
            return []

    if n == 1:
        return response.choices[0].message.content
    return [choice.message.content for choice in response.choices]


def call_reward_model(problem: str, model_answer: str, ground_truth: str):
    question = ""
    if problem:
        try:
            start_index = problem.index("</text>")
            end_index = problem.index("Format your response as follows:")
            question = problem[start_index:end_index].replace("</text>", "").strip()
        except ValueError:
            question = problem.strip()

    orm_response = call_oai_rm_llm(
        system_prompt=GENERAL_ORM_PROMPT,
        prompt=ORM_USER_TEMPLATE.format(problem=question, answer_1=model_answer, answer_2=ground_truth),
        temperature=0.0,
        model_id=os.getenv("VERIFIER_PATH"),
        retry_count=5,
    )
    if "YES" in orm_response:
        return 1.0
    return 0.0


def compute_score(solution_str, ground_truth: list, prompt_str: str = None) -> float:
    def compute_score_single(solution_str, ground_truth) -> float:
        ground_truth = ground_truth.lower()

        retval = 0.
        try:
            string_in_last_boxed = last_boxed_only_string(solution_str)
            if string_in_last_boxed is not None:
                answer = remove_boxed(string_in_last_boxed)
                if is_equiv(answer, ground_truth):
                    retval = 1.
        except Exception as e:
            print(e)
        return retval
    if '</think>' in solution_str:
        solution_str = solution_str.split('</think>')[-1]
    solution_str = solution_str[-300:].lower()
    score = max(compute_score_single(solution_str, gt) for gt in ground_truth)

    if score < 1.0 and os.getenv("LLM_JUDGE") == "Y":
        prompt_str = extract_problem(prompt_str)  # 避免 memory 部分对 llm judge 干扰, 只保留核心 question
        rm_score = max(call_reward_model(prompt_str, solution_str, gt.lower()) for gt in ground_truth)
        # print(f"  RM Score: {rm_score}")
        score = max(score, rm_score)
    return score


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

def is_sub_str(answer, ground_truth, verbose=False):
    if answer is None and ground_truth is None:
        print("WARNING: Both None")
        return True
    if answer is None or ground_truth is None:
        return False

    try:
        ss_ans = strip_string(answer)
        ss_gt = strip_string(ground_truth)
        if verbose:
            print(ss_ans, ss_gt)
        return ss_gt in ss_ans
    except Exception:
        return ground_truth in answer

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    # remove spaces
    string = string.replace(" ", "")

    return string
