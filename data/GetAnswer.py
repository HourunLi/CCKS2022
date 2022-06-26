import json

# fill the filed of json file in the correct form
def getJsFiled(f_r_input, f_r_output, js, cnt):
    while True:
        line_num = f_r_input.tell()
        line = f_r_input.readline()
        line = None if not line else json.loads(line)
        if not line or line["ind"] != cnt:
            f_r_input.seek(line_num)
            break
        label = f_r_output.readline().strip("\n")
        label = int(label)
        if label:
            js[line["relation"]].append(eval(line["pair"]))
    return


"""
This function's purpose is to put the prediction results of two relation matching modules together
And form the final result in its correct form
"""
def getAnswer(tokenizer_result, condition_x_input, condition_x_result, coarse_fine_input, coarse_fine_result, output_file):
    with open(tokenizer_result, 'r', encoding='utf-8') as f_r1, open(condition_x_input, 'r', encoding='utf-8') as f_r2, \
            open(condition_x_result, 'r', encoding='utf-8') as f_r3, open(coarse_fine_input, 'r', encoding='utf-8') as f_r4, \
            open(coarse_fine_result, 'r', encoding='utf-8') as f_r5, open(output_file, 'w', encoding='utf-8') as f_w:
        cnt = 0
        while True:
            js = {"context": "", "question": "", "condition": [], "coarse": [], "fine": [
            ], "condition_coarse": [], "condition_fine": [], "coarse_fine": []}
            raw_data = f_r1.readline()
            if not raw_data:
                break
            cnt += 1
            raw_data = json.loads(raw_data)
            js["context"], js["question"], js["condition"], js["coarse"], js["fine"] = raw_data[
                "context"], raw_data["question"], raw_data["condition"], raw_data["coarse"], raw_data["fine"]
            getJsFiled(f_r2, f_r3, js, cnt)
            getJsFiled(f_r4, f_r5, js, cnt)
            json.dump(js, f_w, ensure_ascii=False)
            f_w.write("\n")
    f_r1.close()
    f_r2.close()
    f_r3.close()
    f_r4.close()
    f_r5.close()
    f_w.close()
    return


def main():
    getAnswer(tokenizer_result="./IntermediateFullData/tokenizer_result.json", condition_x_input="./FullData/predict_for_matchRelat_condition_x.json",
              condition_x_result="../src/relation_condition_x/predict_results.txt", coarse_fine_input="./FullData/predict_for_matchRelat_coarse_fine.json",
              coarse_fine_result="../src/relation_coarse_fine/predict_results.txt", output_file="./answer.json")
    return


if __name__ == "__main__":
    main()
