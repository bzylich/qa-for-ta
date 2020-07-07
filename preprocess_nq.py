from lineflow import TextDataset
import json
from torch.utils.data import DataLoader
from bs4 import BeautifulSoup
import os
import time
import random

long_candidates_per_example = 4

nq_path = "/mnt/nfs/work1/andrewlan/bzylich/datasets/natural_questions/simplified/v1.0-simplified_simplified-nq-train.jsonl"

nq_save_path = "/mnt/nfs/work1/andrewlan/bzylich/datasets/natural_questions/simplified/v1.0-simplified-nq-train-" + str(long_candidates_per_example) + "-cands-per-example.pkl"

# each json has fields: document_text, long_answer_candidates ([{start_token, end_token, top_level} ...]),
# question_text, annotations ([{yes_no_answer, long_answer ({start_token, candidate_index, end_token}),
# short_answers ([{start_token, end_token} ...]), annotation_id} ...]), document_url, example_id

class_split = {False: 0, True: 0}


def process_nq_json(nq_json):
    examples = []
    doc_txt = nq_json["document_text"]
    question_txt = nq_json["question_text"]
    correct_long_answers = set()
    for annotation in nq_json["annotations"]:
        correct_long_answers.add(annotation["long_answer"]["candidate_index"])

    candidates = nq_json["long_answer_candidates"]

    if len(candidates) > long_candidates_per_example:
        if len(correct_long_answers) > 0:
            sampled_candidates = random.sample(range(len(candidates)), long_candidates_per_example)
            if set(sampled_candidates).isdisjoint(correct_long_answers):
                sampled_candidates = sampled_candidates[:-1]
                sampled_candidates.extend(random.sample(correct_long_answers, 1))
        else:
            sampled_candidates = random.sample(range(len(candidates)), long_candidates_per_example)
    else:
        sampled_candidates = range(len(candidates))

    for i in sampled_candidates:  # range(len(candidates)):
        start = candidates[i]["start_token"]
        end = candidates[i]["end_token"]
        # top_level = candidates[i]["top_level"]

        cand_txt = " ".join(doc_txt.split(" ")[start:end])
        soup = BeautifulSoup(cand_txt)
        cand_txt = soup.get_text()

        class_split[i not in correct_long_answers] += 1
        cand_obj = {"question": question_txt, "context": cand_txt, "is_impossible": i not in correct_long_answers}
        print(cand_obj, flush=True)

        examples.append(cand_obj)

    return examples


s_time = time.time()
if os.path.exists(nq_save_path):
    print("File at save path already exists!")
else:
    print("Processing dataset!", flush=True)
    nq = TextDataset(nq_path).map(json.loads).flat_map(process_nq_json)

    print("Testing one batch from dataloader...", flush=True)
    loader = DataLoader(nq, batch_size=128, num_workers=4, shuffle=True)
    it = iter(loader)
    print(next(it))

    print("Saving dataset...", flush=True)
    nq.save(nq_save_path)
    print("Dataset saved to:", nq_save_path, flush=True)
e_time = time.time()
print("Script finished in", e_time - s_time)
print("is_impossible class split:", class_split)
