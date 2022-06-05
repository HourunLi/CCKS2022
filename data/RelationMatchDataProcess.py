import json 
def prepareTrainingData(file_name, output_file_condition_x, output_file_coarse_fine):
    with open(file_name, 'r', encoding='utf-8') as f_r, open(output_file_condition_x, "w", encoding="utf-8") as f_w1, \
        open(output_file_coarse_fine, "w", encoding="utf-8") as f_w2:
        while True:
            line = f_r.readline()
            if not line:
                break
            js = json.loads(line)
            context = js["context"]
            conditions, coarses, fines = js["condition"], js["coarse"], js["fine"]
            condition_coarses, condition_fines, coarse_fines = js["condition_coarse"], js["condition_fine"], js["coarse_fine"]
            relation_set = [relation[1] for relations in [condition_coarses, condition_fines, coarse_fines]
                for relation in relations]

            for (relation1s, relation2s, type) in [(conditions, coarses, "condition_coarse"), (conditions, fines, "condition_fine"), (coarses, fines, "coarse_fine")]:
                matchPotentialRelation(relation1s, relation2s, context, f_w1, f_w2, type, relation_set)
                # for relation1 in relation1s:
                #     for relation2 in relation2s:
                #         pos_tag = [[relation1[1][0], '[unused1]'], [relation1[1][1], '[unused2]'], [relation2[1][0], '[unused3]'], [relation2[1][1], '[unused4]']]
                #         # pos_tag = [[relation1[1][0], '[SEP]'], [relation1[1][1], '[SEP]'], [relation2[1][0], '[SEP]'], [relation2[1][1], '[SEP]']]

                #         pos_tag.sort()
                #         begin, end = pos_tag[0][0], pos_tag[3][0]
                #         context_begin, context_end = 0, len(context)
                #         for i in range(begin, 0, -1):
                #             if context[i] == '\n':
                #                 context_begin = i + 1
                #                 break
                #         for i in range(end, len(context), 1):
                #             if context[i] == '\n':
                #                 context_end = i
                #                 break
                #         js_output["sentence1"] = context[context_begin:pos_tag[0][0]] + pos_tag[0][1] + context[pos_tag[0][0]:pos_tag[1][0]] + pos_tag[1][1] + \
                #             context[pos_tag[1][0]:pos_tag[2][0]] + pos_tag[2][1] + context[pos_tag[2][0]:pos_tag[3][0]] + pos_tag[3][1] + context[pos_tag[3][0]:context_end]
                #         js_output["label"] = 1 if [relation1[1], relation2[1]] in relation_set else 0
                #         js["relation"] = type
                #         if type == "coarse_fine":
                #             json.dump(js_output, f_w2, ensure_ascii=False)
                #             f_w2.write("\n")
                #         else:
                #             json.dump(js_output, f_w1, ensure_ascii=False)
                #             f_w1.write("\n")
    f_r.close()
    f_w1.close()
    f_w2.close()

def matchPotentialRelation(tag1s, tag2s, context, f_w1, f_w2, type, relation_set, index = 0):
    js = {"sentence1": "", "label":0, "pair": "", "relation":"", "ind":0}
    for tag1 in tag1s:
        for tag2 in tag2s:
            pos_tag = [[tag1[1][0], '[unused1]'], [tag1[1][1], '[unused2]'], [tag2[1][0], '[unused3]'], [tag2[1][1], '[unused4]']]
            pos_tag.sort()
            begin, end = pos_tag[0][0], pos_tag[3][0]
            context_begin, context_end = 0, len(context)
            for i in range(begin, 0, -1):
                if context[i] == '\n':
                    context_begin = i + 1
                    break
            for i in range(end, len(context), 1):
                if context[i] == '\n':
                    context_end = i
                    break
            js["sentence1"] = context[context_begin:pos_tag[0][0]] + pos_tag[0][1] + context[pos_tag[0][0]:pos_tag[1][0]] + pos_tag[1][1] + \
                context[pos_tag[1][0]:pos_tag[2][0]] + pos_tag[2][1] + context[pos_tag[2][0]:pos_tag[3][0]] + pos_tag[3][1] + context[pos_tag[3][0]:context_end]
            js["label"] = 1 if [tag1[1], tag2[1]] in relation_set else 0
            js["pair"] = str([[tag1[0], tag2[0]], [tag1[1], tag2[1]]])
            js["relation"] = type
            js["ind"] = index
            if type == "coarse_fine":
                json.dump(js, f_w2, ensure_ascii=False)
                f_w2.write("\n")
            else:
                json.dump(js, f_w1, ensure_ascii=False)
                f_w1.write("\n")
    return

# input: tokenizer prediction result
def predictDataPreprocess(token_predict_result, output_file_condition_x, output_file_coarse_fine):
    relationType = ["condition_coarse", "condition_fine", "coarse_fine"]
    cnt = 0
    with open(token_predict_result, 'r', encoding='utf-8') as f_r, \
        open(output_file_condition_x, "w", encoding="utf-8") as f_w1, open(output_file_coarse_fine, "w", encoding="utf-8") as f_w2:
        while True:
            prediction = f_r.readline()
            if not prediction:
                break
            cnt += 1
            prediction = json.loads(prediction)
            context, conditions, coarses, fines = prediction["context"], prediction["condition"], prediction["coarse"], prediction["fine"]
            matchPotentialRelation(conditions, coarses, context, f_w1, f_w2, relationType[0], [], cnt)
            matchPotentialRelation(conditions, fines, context, f_w1, f_w2, relationType[1], [], cnt)
            matchPotentialRelation(coarses, fines, context, f_w1, f_w2, relationType[2], [], cnt)
    f_r.close()
    f_w1.close()
    f_w2.close()
      
def trainingDataPreprocess():
    # prepareTrainingData(file_name = "./SampleData/train_raw.json", output_file_condition_x = "./SampleData/train_for_matchRelat_condition_x.json",
    #     output_file_coarse_fine = "./SampleData/train_for_matchRelat_coarse_fine.json")
    # prepareTrainingData(file_name = "./SampleData/validation_raw.json", output_file_condition_x = "./SampleData/validation_for_matchRelat_condition_x.json",
    #     output_file_coarse_fine = "./SampleData/validation_for_matchRelat_coarse_fine.json")

    prepareTrainingData(file_name = "./FullData/train_raw.json", output_file_condition_x = "./FullData/train_for_matchRelat_condition_x.json",
        output_file_coarse_fine = "./FullData/train_for_matchRelat_coarse_fine.json")
    prepareTrainingData(file_name = "./FullData/validation_raw.json", output_file_condition_x = "./FullData/validation_for_matchRelat_condition_x.json",
        output_file_coarse_fine = "./FullData/validation_for_matchRelat_coarse_fine.json")

def main():
    trainingDataPreprocess()
    predictDataPreprocess(token_predict_result = "./IntermediateFullData/tokenizer_result.json", \
        output_file_condition_x = "./FullData/predict_for_matchRelat_condition_x.json", output_file_coarse_fine = "./FullData/predict_for_matchRelat_coarse_fine.json")

if __name__ == "__main__":
    main()
