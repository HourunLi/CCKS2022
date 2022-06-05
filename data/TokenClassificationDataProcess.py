import json

def sequenceLabeling(tag, slice, label):
    for item in slice:
        begin, end = item[1][0], item[1][1]
        for i in range(begin, end):
            tag[i] = label
    return

# tag : 0 for None, 1 for condition, 2 for coarse, 3 for fine, 4 for question
"""
 This function differs with the next function below. The difference lies in the spilt ways.
 prepareTrainingData splits the context into several sentences and the word in sentence may be ignored when bert encodes.
 This function is deprecated but yet not removed.
"""
def prepareTrainingData(file_name, output_file):
    js_output = {"tokens": [], "tags": []}
    # js_output = {"tokens": []}
    with open(file_name, 'r', encoding='utf-8') as f_r, open(output_file, "w", encoding="utf-8") as f_w:
        cnt = 0
        while True:
            cnt += 1
            line = f_r.readline()
            if not line:
                break
            js = json.loads(line)
            context = js["context"]
            partitions, context_split, tags = [], [js["question"], '[SEP]'], [4, 0]
            for i, slices in enumerate([js["condition"], js["coarse"], js["fine"]]):
                # print(i, slices)
                for slice in slices:
                    begin, end = slice[1][0], slice[1][1]
                    partitions.append([begin, end, i+1])
            partitions.sort()

            index = 0
            for partition in partitions:
                if index < partition[0]:
                    context_split.append(context[index: partition[0]].replace(" ", "[PAD]").replace("\n", "[PAD]"))
                    tags.append(0)
                    index = partition[0]
                begin = min(index, partition[0])
                context_split.append(context[begin: partition[1]].replace(" ", "[PAD]").replace("\n", "[PAD]"))
                tags.append(partition[2])
                index = max(index, partition[1])

            if index < len(context):
                context_split.append(context[index:])
                tags.append(0)
                index = len(context)

            if(len(context_split) != len(tags)):
                raise NameError(
                    "Context partition error: length of tokens and tags are not the same")

            js_output["tokens"] = context_split
            js_output["tags"] = tags
            # print(json.dumps(js_output))
            json.dump(js_output, f_w, ensure_ascii=False)
            f_w.write("\n")
    f_r.close()
    f_w.close()

# label_to_id = {
#     "None":0,
#     "B-condition":1,
#     "I-condition":2,

# }
"""
 tag: 0 for None, 1 for condition, 2 for condition, 3 for coarse, 4 for [SEP].
 flag: 0 for prediction, 1 for traning or validation
 This function differs with the last function above. The difference lies in the spilt ways.
 prepareTrainingData_singleWord splits the context into single words and makes sure the word won't be ignore when bert encodes.
"""
def prepareTrainingData_singleWord(file_name, output_file, flag = 1):
    js_output = {"tokens": [], "tags": []}
    with open(file_name, 'r', encoding='utf-8') as f_r, open(output_file, "w", encoding="utf-8") as f_w:
        cnt = 0
        while True:
            cnt += 1
            line = f_r.readline()
            if not line:
                break
            js = json.loads(line)
            context = js["context"]
            # Because " " and "\n" won't be encoded in bert
            # so we must substitute the characters in the context
            # in order to align the tag and char
            context_split = ['[PAD]' if ch == ' ' else ch for ch in context]
            context_split = ['[unused1]' if ch == '\n' else ch for ch in context_split]
            # 0 for question, 8 for [SEP]
            tags = [0, 4] + [0] * len(js["context"])
            if flag:
                for tag, slices in enumerate([js["condition"], js["coarse"], js["fine"]]):
                    for slice in slices:
                        begin, end = slice[1][0], slice[1][1]
                        for i in range(begin+2, end+2):
                            tags[i] = tag + 1
                        # if begin+1 != end:
                        #     tags[begin+2] = 2*tag + 2
            context_split = [js["question"].strip(" \n").strip("？?"), '[SEP]'] + context_split
            # we provide unique tags fot speicial but common characters
            # We hope to provide the bert model with more info as many as possible 
            # special_tokens = {'[SEP]':7, "[unused1]":8, "。":9}
            # for i, ch in enumerate(context_split):
            #     if ch in special_tokens.keys() and not tags[i]:
            #         tags[i] = special_tokens[ch]

            # Make sure the length of context chars and tags have the same length
            assert len(tags) == len(context_split), "the length of tags doesn't equal to that of context."
            js_output["tokens"] = context_split
            js_output["tags"] = tags
            json.dump(js_output, f_w, ensure_ascii=False)
            f_w.write("\n")
    f_r.close()
    f_w.close()

boundary_tokens = ['[CLS]', '[SEP]', '[PAD]', '[unused1]',",", '，','、','。','.','-','？','；', '（','）']
def processPredictLine(line, js):
    label_list = {1:"condition", 2:"coarse", 3:"fine"}
    special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[unused1]', '，', '、', '；', '。', ',', '.', '及','和' ,'是','或', '还','(', ')','（','）']
    context, labels = line[0], line[1]
    begin, end = 0, len(context) - 1
    # find the first [SEP] position, where context begins
    for ind, word in enumerate(context):
        if word == '[SEP]':
            begin = ind+1
            break
    
    string = ""
    l = begin
    # use the double pointer method to extract the desired answer: condition, coarse, fine
    while l < end:
        curr_label = labels[l]
        if curr_label <= 0 or curr_label >= 4:
            l += 1
            continue
        r = l
        while labels[r]  == curr_label and r < end:
            r += 1
        tmp_l, tmp_r = l, r
        # special characters at head or tail should not be included.
        while(context[tmp_l] in special_tokens):
            tmp_l += 1
        while(context[tmp_r-1] in special_tokens):
            tmp_r -= 1
        string = "".join(context[tmp_l: tmp_r])
        # condition ansers should at least be 4 characters long
        if curr_label != 1 and string or curr_label == 1 and len(string) > 3:
            js[label_list[curr_label]].append([string, [tmp_l-begin, tmp_r-begin]])
        l = r
    # further process the condition answer 
    # we need to adjust the boundary of condition answer
    conditions = js["condition"]
    tuning_conditions = []
    for condition in conditions:
        a, b = condition[1][0], condition[1][1]
        while a > 0:
            if context[a-1+begin] in boundary_tokens:
                break
            a -=1
        while b <= end:
            if context[b+begin] in boundary_tokens:
                break
            b += 1
        tmp = ["".join(context[a+begin: b+begin]), [a, b]]
        if tmp not in tuning_conditions:
            tuning_conditions.append(tmp)
    js["condition"] = tuning_conditions
    # further process the coarse condition answer
    # we need to merge the adjecent but seperated and short coarse, which is more like to be a whole entity
    coarse = js["coarse"]
    tuning_coarse = []
    # the order is from small to large
    # still use the double pointer method to merge them
    coarse_len = len(coarse)
    l = 0
    while l < coarse_len:
        r = l +1
        pause = False
        while r < coarse_len:
            pre_end, curr_begein = coarse[r-1][1][1], coarse[r][1][0]
            if curr_begein > pre_end + 1:
                pause = True
                break
            for i in range(pre_end, curr_begein):
                if context[i+begin] in special_tokens:
                    pause = True
                    # print(context[i+begin], end= " ")
                    break
            if pause:
                break
            r += 1
        if coarse[r-1][1][1] - coarse[l][1][0] >=2:
            tmp = ["".join(context[coarse[l][1][0]+begin: coarse[r-1][1][1]+begin]), [coarse[l][1][0], coarse[r-1][1][1]]]
            if r > l + 1:
                for i in range(l, r):
                    print(coarse[i], end = " ")
                print(tmp)
            tuning_coarse.append(tmp)
        l = r
    js["coarse"] = tuning_coarse
    # further process the fine condition answer
    # we need to merge the adjecent but seperated and short fine, which is more like to be a whole entity
    fine = js["fine"]
    tuning_fine = []
    # the order is from small to large
    # still use the double pointer method to merge them
    fine_len = len(fine)
    l = 0
    while l < fine_len:
        r = l +1
        pause = False
        while r < fine_len:
            pre_end, curr_begein = fine[r-1][1][1], fine[r][1][0]
            if curr_begein > pre_end + 1:
                pause = True
                break
            for i in range(pre_end, curr_begein):
                if context[i+begin] in special_tokens:
                    pause = True
                    # print(context[i+begin], end= " ")
                    break
            if pause:
                break
            r += 1
        if fine[r-1][1][1] - fine[l][1][0] >=2:
            tmp = ["".join(context[fine[l][1][0]+begin: fine[r-1][1][1]+begin]), [fine[l][1][0], fine[r-1][1][1]]]
            if r > l + 1:
                for i in range(l, r):
                    print(fine[i], end = " ")
                print(tmp)
            tuning_fine.append(tmp)
        l = r
    js["fine"] = tuning_fine
    return

def getPredictResult(raw_data, tokenizer_predictions_file, output_file):
    with open(raw_data, 'r', encoding='utf-8') as f_r1, open(tokenizer_predictions_file, 'r', encoding='utf-8') as f_r2, open(output_file, "w", encoding="utf-8") as f_w:
        raw_data = f_r1.readline()
        line = f_r2.readline()
        while raw_data and line:
            line = eval(line)
            raw_data = json.loads(raw_data)
            js = {"context":"", "question":"", "condition":[], "coarse":[], "fine":[]}
            js["context"], js["question"]= raw_data["context"], raw_data["question"]
            processPredictLine(line, js)
            json.dump(js, f_w, ensure_ascii=False)
            f_w.write("\n")
            raw_data = f_r1.readline()
            line = f_r2.readline()

def trainingDataPreprocess():
    # prepareTrainingData_singleWord(file_name = "./SampleData/train_raw.json", output_file = "./SampleData/train_for_tokClasfy.json")
    # prepareValidorPredictData(file_name = "./SampleData/validation_raw.json", output_file = "./SampleData/validation_for_tokClasfy.json", flag = 1)
    # prepareTrainingData_singleWord(file_name = "./SampleData/validation_raw.json", output_file = "./SampleData/validation_for_tokClasfy.json")
    # prepareTrainingData_singleWord(file_name = "./SampleData/predict_raw.json", output_file = "./SampleData/predict_for_tokClasfy.json", flag = 0)
    # prepareTrainingData(file_name = "./SampleData/predict_raw.json", output_file = "./SampleData/predict_for_tokClasfy.json")

    # prepareTrainingData(file_name = "./FullData/train_raw.json", output_file = "./FullData/train_for_tokClasfy.json")
    # prepareTrainingData(file_name = "./FullData/validation_raw.json", output_file = "./FullData/validation_for_tokClasfy.json")

    prepareTrainingData_singleWord(file_name = "./FullData/train_raw.json", output_file = "./FullData/train_for_tokClasfy.json")
    prepareTrainingData_singleWord(file_name = "./FullData/validation_raw.json", output_file = "./FullData/validation_for_tokClasfy.json")
    prepareTrainingData_singleWord(file_name = "./FullData/predict_raw.json", output_file = "./FullData/predict_for_tokClasfy.json", flag = 0)

def predictDataPostprocess():
    getPredictResult(raw_data = "./FullData/predict_raw.json", tokenizer_predictions_file = "../src/tokenizer/predictions.txt", output_file = "./IntermediateFullData/tokenizer_result.json")

def main():
    trainingDataPreprocess()
    predictDataPostprocess()

if __name__ == "__main__":
    main()
