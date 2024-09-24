import json
import time
from llm import LLM_Client
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

llm_client = LLM_Client()


def process_item(item):
    question = "Question_type: " + item["problem_type"] + "\nQuestion: " + item["question"]
    try:
        code, answer = llm_client.integrate_get_answer(content=question)
    except:
        code = ""
        answer = "1"

    result_item = {
        "ID": item["ID"],
        "question": question,
        "question_type": item["problem_type"],
        "code": code,
        "answer": answer,
    }
    return result_item


def save_results(results, output_file):

    results_sorted = sorted(results, key=lambda x: x["ID"])
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(results_sorted, outfile, ensure_ascii=False, indent=4)


def process_json(input_file, output_file, num_threads):
    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    result = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for item in data:
            future = executor.submit(process_item, item)
            futures.append(future)
            time.sleep(0.3)
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            result_item = future.result()
            result.append(result_item)

            # Save interim results
            if (i + 1) % 2 == 0:
                save_results(result, output_file)

    # Save the final result
    save_results(result, output_file)


if __name__ == "__main__":
    input_file = "database/Final_TestSet/Final_TestSet.json"
    output_file = "submit/xiatian_002_result.json"
    num_threads = 2

    process_json(input_file, output_file, num_threads)
