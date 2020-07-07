import drqa.tokenizers as dk
import drqa.retriever as ret
# from drqa import pipeline
import drqa.reader as dr
import datetime
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from adapter_bert import AdapterBert, setup_adapters
from transformers import BertTokenizer
from typing import Dict

import prettytable

corenlp_path = '/home/bzylich/DrQA/DrQA/data/corenlp/*'  # CoreNLP path (should be installed along with DrQA- included in those instructions: https://github.com/facebookresearch/DrQA)
tfidf_path = "/home/bzylich/ai-as-ta/drqa/with_posts/docs_with_posts-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz"  # tfidf model from processing step
model_path = '/home/bzylich/DrQA/DrQA/data/reader/multitask.mdl'  # DrQA pretrained model from https://github.com/facebookresearch/DrQA

docs_json_path = "/home/bzylich/ai-as-ta/data/materials_2018/docs_with_posts.json"  # json file containing all docs from preprocessing step

db_path = "/home/bzylich/ai-as-ta/drqa/with_posts/docs_with_posts.db"  # db file from processing step

timestamps_path = "/home/bzylich/ai-as-ta/data/materials_2018/file_timestamps_with_posts.csv"  # timestamps file from preprocessing step
forum_data_path = "/home/bzylich/ai-as-ta/src/question_types/Spring 2018 Data Deidentified with Question Type.xlsx"  # path to excel file containing forum questions (currently uses Question Type field to identify logistics questions)

results_csv_path = "./two_step/physics-logistic-question-results_two-step_normalize_question-only_no-question-answers.csv"  # output containing csv of top 5 potential answers to each question


# answerability files

model_name = "bert-base-uncased"
load_name = "bert-base-uncased_0.1_32_adapter_8_epochs_20_SQuAD_transfer_NQ_4-cands-per-example__ckpt_epoch_19.ckpt"


NUM_LABELS = 2
BATCH_SIZE = 32
MAX_LEN = 300
load_adapter_size = 8

# settings

num_predictions = 5
GROUP_LENGTH = 500  # character cutoff for a new paragraph


print("starting drqa test...", flush=True)
dk.set_default('corenlp_classpath', corenlp_path)

retriever = ret.get_class('tfidf')(tfidf_path=tfidf_path)

print("Retriever model loaded", flush=True)

dr.set_default('model', model_path)
reader = dr.Predictor(model_path, "corenlp", normalize=True)

print("drqa reader loaded", flush=True)


# set up answerability classifier
print("loading answerability classifier...", flush=True)
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case="uncased" in model_name)  # , cache_dir=cache_directory)


class Model(pl.LightningModule):

    def __init__(self):
        super(Model, self).__init__()
        setup_adapters(load_adapter_size)
        model = AdapterBert.from_pretrained(model_name, num_labels=NUM_LABELS)  # , cache_dir=cache_directory)
        # model = BertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS, cache_dir=cache_directory)
        self.model = model


pretrained_model = Model()

checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
pretrained_model.load_state_dict(checkpoint['state_dict'])

pretrained_model.zero_grad()
pretrained_model.eval()
pretrained_model.freeze()
torch.set_grad_enabled(False)

print("answerability classifier loaded successfully", flush=True)

#


docs_txt = {}
with open(docs_json_path, encoding='utf-8') as docs_text:
    for line in docs_text:
        line = eval(line)
        docs_txt[line["id"]] = line["text"]

docs_timestamps = {}
with open(timestamps_path, encoding="utf-8") as timestamps_file:
    for line in timestamps_file:
        parts = line.strip().rsplit(",", 1)
        date_txt = parts[1]
        if ":" in date_txt:
            d_date = datetime.datetime.strptime(date_txt, "%Y-%m-%d %H:%M:%S")
        else:
            d_date = datetime.datetime.strptime(date_txt, "%Y-%m-%d")

        doc_name = parts[0]
        if "/" in doc_name:
            doc_name = doc_name.split("/")[-1]
        if "\\" in doc_name:
            doc_name = doc_name.split("\\")[-1]
        doc_name = doc_name.rsplit(".", 1)[0]

        docs_timestamps[doc_name] = d_date

print(docs_timestamps)

forum_data = pd.read_excel(forum_data_path)

threads = {}
for index, row in forum_data.iterrows():
    thread_id = row["Post Number"]
    if thread_id not in threads:
        threads[thread_id] = []

    threads[thread_id].append(row)

for t in threads:
    threads[t] = sorted(threads[t], key=lambda p: datetime.datetime.strptime(p["Created At"], "%Y-%m-%d %H:%M:%S %Z"))


def closest_docs(query, k=1):
    """Closest docs by dot product between query and documents
    in tfidf weighted word vector space.
    """

    try:
        spvec = retriever.text2spvec(query)
    except Exception:
        return [], []
    res = spvec * retriever.doc_mat
    query_magnitude = np.sqrt(spvec.multiply(spvec).sum())
    doc_magnitude = np.array((retriever.doc_mat).multiply(retriever.doc_mat).sum(axis=0)).flatten()
    doc_magnitude = np.sqrt(doc_magnitude)[res.indices]
    denominator = np.array(query_magnitude * doc_magnitude).flatten()
    res.data /= denominator

    if len(res.data) <= k:
        o_sort = np.argsort(-res.data)
    else:
        o = np.argpartition(-res.data, k)[0:k]
        o_sort = o[np.argsort(-res.data[o])]

    doc_scores = res.data[o_sort]
    doc_ids = [retriever.get_doc_id(i) for i in res.indices[o_sort]]
    return doc_ids, doc_scores


def rank_docs(query, k=5, date=None, post_num=None):
    doc_names = []
    doc_scores = []
    doc_dates = []
    k_prime = 0
    curr_doc_txts = set()

    if post_num is not None:
        post_num_name = "post_" + str(post_num)

    while len(doc_names) < k:
        k_prime += k
        d_names, d_scores = closest_docs(query, k_prime)
        if len(d_names) == 0:
            return [], [], []
        for i, d_name in enumerate(d_names):
            if post_num is not None:
                if d_name == post_num_name:
                    continue

            # do not use forum questions as answers
            if "post_" in d_name:
                post_row = forum_data.iloc[int(d_name.split("post_")[1])]
                if "question" in post_row["Part of Post"]:
                    continue
            # end

            curr_txt = docs_txt[d_name]
            if curr_txt in curr_doc_txts:
                continue
            curr_doc_txts.add(curr_txt)

            d_date = docs_timestamps[d_name]
            if date is not None:
                if d_date < date and d_name not in doc_names:
                    doc_names.append(d_name)
                    doc_scores.append(d_scores[i])
                    doc_dates.append(d_date)

                    if len(doc_names) >= k:
                        return doc_names, doc_scores, doc_dates
            else:
                doc_names.append(d_name)
                doc_scores.append(d_scores[i])
                doc_dates.append(d_date)

                if len(doc_names) >= k:
                    return doc_names, doc_scores, doc_dates

    return doc_names, doc_scores, doc_dates


def provide_answer(document, question, candidates=None, top_n=3):
    predictions = reader.predict(document, question, candidates, top_n)
    table = prettytable.PrettyTable(['Rank', 'Span', 'Score'])
    for i, p in enumerate(predictions, 1):
        table.add_row([i, p[0], p[1]])
    return predictions


# for answerability classifier
def preprocess(tokenizer: BertTokenizer, x: Dict) -> Dict:
    # Given two sentences, x["string1"] and x["string2"], this function returns BERT ready inputs.
    inputs = tokenizer.encode_plus(
            x["question"],
            x["context"],
            add_special_tokens=True,
            max_length=MAX_LEN,
            )

    # First `input_ids` is a sequence of id-type representation of input string.
    # Second `token_type_ids` is sequence identifier to show model the span of "string1" and "string2" individually.
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)

    # BERT requires sequences in the same batch to have same length, so let's pad!
    padding_length = MAX_LEN - len(input_ids)

    pad_id = tokenizer.pad_token_id
    input_ids = input_ids + ([pad_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([pad_id] * padding_length)

    # Super simple validation.
    assert len(input_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(input_ids), MAX_LEN)
    assert len(attention_mask) == MAX_LEN, "Error with input length {} vs {}".format(len(attention_mask), MAX_LEN)
    assert len(token_type_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(token_type_ids), MAX_LEN)

    # Convert them into PyTorch format.
    input_ids = torch.tensor([input_ids])
    attention_mask = torch.tensor([attention_mask])
    token_type_ids = torch.tensor([token_type_ids])

    # DONE!
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    }


delimiter = "|--*--|"
with open(results_csv_path, "w", encoding="utf-8") as out_file:
    for index, row in forum_data.iterrows():
        post_type = row["Forum Role"]
        if post_type in ["p_q", "f_q"]:
            question_type = row["Question Type"]
            if question_type is not None and question_type == "L":
                question_text = str(row["Subject"]).replace("\n", " ") + " " + str(row["Submission HTML Removed"]).replace("\n", " ")
                question_timestamp = str(row["Created At"])
                print("-----------------------------------------------------------", flush=True)
                print("Question:", question_text, flush=True)
                question_date = datetime.datetime.strptime(question_timestamp, "%Y-%m-%d %H:%M:%S %Z")
                print("Question timestamp:", question_timestamp, "date:", question_date, flush=True)
                print("Question type:", question_type, "Post number:", index, flush=True)

                potential_answers = []
                for other_post in threads[row["Post Number"]]:
                    other_date = datetime.datetime.strptime(other_post["Created At"], "%Y-%m-%d %H:%M:%S %Z")
                    if other_date > question_date:
                        other_type = other_post["Forum Role"]
                        if "_a" in other_type:
                            other_text = other_post["Submission HTML Removed"].replace("\n", " ")
                            if other_text not in potential_answers:
                                potential_answers.append(other_text)

                out_file.write(str(post_type) + delimiter + str(question_type) + delimiter + str(question_date) + delimiter + str(question_text) + delimiter)

                doc_names, doc_scores, doc_dates = rank_docs(question_text, k=5, post_num=index)#, date=question_date)  # TO RESTRICT DATE
                print(len(doc_names), "documents retrieved", flush=True)

                # COMMENT TO KEEP SUBJECT IN QUESTION TEXT
                question_text = str(row["Submission HTML Removed"]).replace("\n", " ")
                if question_text == "nan":
                    question_text = str(row["Subject"]).replace("\n", " ").strip()
                # COMMENT

                doc_index = {}
                all_predictions = []

                paragraph_txts = set()
                for i in range(len(doc_names)):
                    doc_index[doc_names[i]] = {"score": doc_scores[i], "date": doc_dates[i]}

                    # get text of document and split into paragraphs
                    curr_txt = docs_txt[doc_names[i]]

                    paragraphs = []
                    curr_paragraph = []
                    curr_len = 0
                    for split in curr_txt.split("\n"):
                        split = split.strip()
                        if len(split) == 0:
                            continue

                        if len(curr_paragraph) > 0 and curr_len + len(split) > GROUP_LENGTH:
                            new_paragraph = " ".join(curr_paragraph)
                            if new_paragraph not in paragraph_txts:
                                paragraphs.append(new_paragraph)
                                paragraph_txts.add(new_paragraph)

                            curr_paragraph = []
                            curr_len = 0
                        curr_paragraph.append(split)
                        curr_len += len(split)
                    if len(curr_paragraph) > 0:
                        paragraphs.append(" ".join(curr_paragraph))

                    best_score = 0
                    best_prediction = None
                    for paragraph in paragraphs:
                        predictions = provide_answer(paragraph, question_text, top_n=1)
                        predictions = list(map(lambda e: (e[0], e[1], doc_names[i], paragraph), predictions))

                        for p in predictions:
                            if p[1] > best_score:
                                best_score = p[1]
                                best_prediction = p

                    all_predictions.append(best_prediction)

                    # answer_text = p['span']
                    # answer_score = p['span_score']
                    # out_file.write(str(answer_text) + delimiter + str(answer_score) + delimiter)
                    #
                    # doc_id = p['doc_id']
                    # doc_score = p['doc_score']
                    # doc_date = docs_timestamps[doc_id]
                    # context = p['context']['text'].replace("\n", " ")
                    #
                    # out_file.write(str(doc_id) + delimiter + str(doc_score) + delimiter + str(doc_date) + delimiter + str(context) + delimiter)

                    # if doc_date >= question_date:
                    #     doc_available = False
                    #     print("****", doc_id, "not available when question was asked (question:", question_date, "doc:", doc_date, ")", flush=True)
                    # else:
                    #     doc_available = True
                    #
                    # out_file.write(str(doc_available) + delimiter)


                all_predictions = sorted(all_predictions, key=lambda r: r[1], reverse=True)[:num_predictions]

                table = prettytable.PrettyTable(['Rank', "Answer", "Doc", "Answer Score", 'Doc Score', 'Doc Date'])

                if len(all_predictions) == 0:
                    na_token = "nan"
                    for i in range(num_predictions):
                        out_file.write(na_token + delimiter + na_token + delimiter)
                        out_file.write(na_token + delimiter + na_token + delimiter + na_token + delimiter + na_token + delimiter)
                        out_file.write(na_token + delimiter)
                else:
                    for i, p in enumerate(all_predictions, 1):
                        rank = i
                        ans = p[0]
                        doc_name = p[2]
                        ans_score = p[1]
                        context = p[3]

                        preprocessed = preprocess(tokenizer, {"question": question_text, "context": context})
                        input_ids = preprocessed["input_ids"]
                        attention_mask = preprocessed["attention_mask"]
                        token_type_ids = preprocessed["token_type_ids"]

                        logits = pretrained_model.model(
                            input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask
                        )
                        answerability_score = torch.nn.functional.softmax(logits[0][0], dim=0)[1]

                        doc_score = doc_index[doc_name]["score"]
                        doc_date = doc_index[doc_name]["date"]
                        table.add_row([rank, ans, doc_name, ans_score, doc_score, doc_date])

                        out_file.write(str(ans) + delimiter + str(ans_score) + delimiter)
                        out_file.write(str(doc_name) + delimiter + str(doc_score) + delimiter + str(doc_date) + delimiter + str(context) + delimiter)
                        doc_available = (doc_date < question_date)
                        out_file.write(str(doc_available) + delimiter + str(answerability_score) + delimiter)
                print(table, flush=True)

                for i, (answer, score, d_name, paragraph) in enumerate(all_predictions, 1):
                    doc_score = doc_index[d_name]["score"]

                    print(i, "--- context:", paragraph, flush=True)

                out_file.write(str(potential_answers) + "\n")
