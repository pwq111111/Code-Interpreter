import json
from llm import LLM_Client
from tqdm import tqdm

llm_client = LLM_Client()


def process_json(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    result = []
    for i, item in enumerate(tqdm(data, total=len(data))):
        question = "Question_type: " + item["problem_type"] + "\nQuestion: " + item["question"]

        code, answer = llm_client.integrate_get_answer(content=question)

        result_item = {
            "ID": item["ID"],
            "question": question,
            "question_type": item["problem_type"],
            "code": code,
            "answer": answer,
        }
        result.append(result_item)

        # Save interim results
        if (i + 1) % 5 == 0:

            with open(output_file, "w", encoding="utf-8") as outfile:
                json.dump(result, outfile, ensure_ascii=False, indent=4)

        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(result, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    input_file = "database/Final_TestSet/Final_TestSet.json"
    output_file = "submit/xiatian_001_result.json"

    process_json(input_file, output_file)
