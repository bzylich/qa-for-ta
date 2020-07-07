import fitz
import glob
import os
import datetime
import numpy as np

import pandas as pd
import docx2txt
import html2text

top_dir = "../data/materials_2018/"  # directory from which to recursively search downward for documents
processed_dir = top_dir + "txt_files_with_posts/"  # directory where preprocessed files will be stored
forum_data_path = "./question_types/Spring 2018 Data Deidentified with Question Type.xlsx"  # used for special case of forum post excel file
out_json_name = "docs_with_posts.json"  # this is the main output from this script, used in the next step of database creation
timestamps_filepath = "file_timestamps_with_posts.csv"  # this file is also used in the question answering system, includes file timestamps for filtering based on when files were available

GROUP_LENGTH = 220  # character cutoff for a new paragraph


def write_file_timestamp(out_file, cur_file, date=None, original_name=None):
    if date is not None:
        creation_time = date
    else:
        if original_name is not None:
            creation_time = os.stat(original_name).st_ctime
        else:
            creation_time = os.stat(cur_file).st_ctime
        # print(creation_time)
        creation_time = str(datetime.date.fromtimestamp(creation_time))
    out_file.write(cur_file + "," + creation_time + "\n")


forum_data = pd.read_excel(forum_data_path)

paragraph_txts = set()


def split_to_paragraphs(curr_txt):
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

    return paragraphs


def write_to_file(txt, doc_name, doc_date=None, original_name=None, post_subject=None):
    assert doc_date is not None or original_name is not None
    pars = split_to_paragraphs(txt)

    i = 0
    for par in pars:
        if post_subject is not None:
            par = post_subject + ".\t" + par

        doc_name_temp = doc_name.replace(".txt", "_" + str(i) + ".txt")
        new_filename = processed_dir + doc_name_temp

        # if not os.path.exists(new_filename):
        write_file_timestamp(timestamps_file, doc_name_temp, date=doc_date, original_name=original_name)
        with open(new_filename, "wb") as txt_file:
            txt_file.write(par.encode('utf-8', 'ignore'))

        i += 1


post_lengths = []
with open(top_dir + timestamps_filepath, "w", encoding='utf-8') as timestamps_file:
    print("start")

    for index, row in forum_data.iterrows():
        post_text = str(row["Submission HTML Removed"]).replace("\n", " ")
        text_len = len(post_text)
        print(text_len)
        post_lengths.append(text_len)
        post_date = str(datetime.datetime.strptime(row["Created At"], "%Y-%m-%d %H:%M:%S %Z"))

        try:
            post_filename = "post_" + str(index) + ".txt"
            write_to_file(post_text, post_filename, doc_date=post_date, post_subject=str(row["Subject"]))
        except Exception:
            print("skipping")
            pass

    print("Average post length:", np.mean(post_lengths), "Maximum post length:", np.max(post_lengths))

    for pdf_name in glob.glob(top_dir + "**/*.pdf", recursive=True):
        try:
            print(pdf_name)
        except Exception:
            pass

        # try:
        pdf = fitz.open(pdf_name)

        doc_name = pdf_name.replace('.pdf', '.txt').split("\\")[-1]

        pdf_text = ""
        for i in range(pdf.pageCount):
            pdf_page = pdf[i]
            pdf_text += pdf_page.getText("text") + " "  # "html"

        write_to_file(pdf_text, doc_name, original_name=pdf_name)

        # except Exception:
        #     print("skipping")
        #     pass

    h = html2text.HTML2Text()
    h.ignore_links = True
    for html_name in glob.glob(top_dir + "**/*.html", recursive=True):
        # write_file_timestamp(timestamps_file, html_name)
        try:
            print(html_name)
        except Exception:
            pass

        with open(html_name) as html_file:
            html_txt = h.handle(html_file.read())

        # print(html_txt)
        html_txt = html_txt.split("=====================================================================================  \n|\n\n")[1:]
        email_index = 0
        for email in html_txt:
            # print(email)
            email_name = email.split("\n\nby")[0]
            email_parts = email.split("---|---")[0].split("\n\nby")[-1].strip()
            # print(email_parts)
            email_author = email_parts.split(" - ")[0].strip()
            email_date = email_parts.split(" - ")[1].split(", ", 1)[1]
            # print(email_date)
            email_date = str(datetime.datetime.strptime(email_date, "%B %d, %Y, %I:%M %p"))
            # print(email_date)
            email_content = email.split("---|---")[1].split("|",1)[1].strip()

            doc_name = str(email_index)+"_"+email_name+"_"+email_author+'.txt'
            # print(html_name)
            # print(email_date)
            write_to_file(email_content, doc_name, doc_date=email_date)

            email_index += 1

    for docx_name in glob.glob(top_dir + "**/*.docx", recursive=True):
        try:
            print(docx_name)
        except Exception:
            pass

        try:
            docx_txt = docx2txt.process(docx_name)
            doc_name = docx_name.replace('.docx', '.txt').split("\\")[-1]

            write_to_file(docx_txt, doc_name, original_name=docx_name)
        except Exception:
            print("skipping")
            pass

    for xlsx_name in glob.glob(top_dir + "**/*.xlsx", recursive=True):
        try:
            print(xlsx_name)
        except Exception:
            print("skipping")
            pass

        try:
            xlsx = pd.read_excel(xlsx_name)
            xlsx_txt = xlsx.to_string()

            doc_name = xlsx_name.replace('.xlsx', '.txt').split("\\")[-1]

            write_to_file(xlsx_txt, doc_name, original_name=xlsx_name)
        except Exception:
            print("skipping")
            pass

    with open(top_dir + out_json_name, "w", encoding="utf-8") as docs_file:
        for txt_name in glob.glob(processed_dir + "*.txt"):
            short_name = txt_name.replace(".txt", "").split("\\")[-1]

            with open(txt_name, encoding="utf-8") as doc_txt_file:
                docs_file.write("{\"id\": \"" + short_name + "\", \"text\": \"" + doc_txt_file.read().replace("\t", " ").replace("\\", "\\\\").replace("\n", "\\n").replace("\"", "\\\"") + "\"}\n")

    txt_lengths = []
    for txt in paragraph_txts:
        txt_lengths.append(len(txt))

    print("min:", np.min(txt_lengths), "avg:", np.mean(txt_lengths), "max:", np.max(txt_lengths))

    print("end")
