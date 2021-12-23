

import numpy
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymysql


host = 'database-skku.c6dzc5dnqf69.ap-northeast-2.rds.amazonaws.com'
iid ='admin'
pw = 'tjdrbsrhkseo123'
db_name = 'dongwan'
conn = pymysql.connect(host=host, user= iid, password=pw, db=db_name, charset='utf8')

curs = conn.cursor(pymysql.cursors.DictCursor)

sql1 = """SELECT * FROM public.clinical_disease"""
sql2 = """SELECT * FROM medii.TotalDisease"""
sql3 = """SELECT * FROM public.doctor_total_disease"""
sql4 = """SELECT * FROM medii.cris_dataset"""


curs.execute(sql1)
rows = curs.fetchall()
clinical_disease = pd.DataFrame(rows)
clinical_disease = clinical_disease.fillna("")


curs.execute(sql2)
rows = curs.fetchall()
diseasecode_disease = pd.DataFrame(rows)


curs.execute(sql3)
rows = curs.fetchall()
doctor_totaldisease = pd.DataFrame(rows)
doctor_totaldisease = doctor_totaldisease.fillna("")


curs.execute(sql4)
rows = curs.fetchall()
doctor_clinical = pd.DataFrame(rows)
doctor_clinical = doctor_clinical.fillna("")


def paper_score(input, w1, w2):
    doctor_paper_data = doctor_totaldisease.copy()

    def preprocess(text):
        text = text.replace('.', "dot")

        return text

    def overlap_paper(text):
        paper_overlap = 0
        papers = text.split('/ ')
        for paper in papers:
            paper = paper.split(', ')
            if all(temp in paper for temp in std):
                paper_overlap += 1

        return paper_overlap

    def overlap_keyword(text):
        words_count = {}

        text = text.replace('/ ', ', ')
        words = text.split(', ')
        word_target = set(words)
        add_keyword = set(std) & word_target

        for word in words:
            if word in words_count:
                words_count[word] += 1
            else:
                words_count[word] = 1

        sorted_words = sorted([(k, v) for k, v in words_count.items()], key=lambda word_count: -word_count[1])
        keyword = [w for w in sorted_words if w[0] in add_keyword]
        if (len(keyword) >= 5):
            keyword = keyword[0:5]

        return keyword

    target_input = preprocess(input)
    target_name = list(doctor_paper_data['name_kor'])
    target_index = len(target_name)
    target_name.append('target')

    text = list(doctor_paper_data['paper_disease_all'])
    target_text = [preprocess(t) for t in text]
    target_text.append(target_input)

    doctors = pd.DataFrame({'name': target_name,
                            'text': target_text})

    tfidf_vector = TfidfVectorizer(min_df=3, max_features=6000)
    tfidf_matrix = tfidf_vector.fit_transform(doctors['text']).toarray()

    cosine_sim = cosine_similarity(tfidf_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, columns=doctors.name)
    cosine_sim_df.head()

    temp = cosine_sim_df['target'][0:target_index]
    doctor_paper_data['cosine_simil'] = temp

    std = input.split(', ')

    doctor_paper_data['keyword_paper'] = doctor_paper_data.apply(lambda x: overlap_keyword(x['paper_disease_all']),
                                                                 axis=1)
    doctor_paper_data['overlap_paper'] = doctor_paper_data.apply(lambda x: overlap_paper(x['paper_disease_all']),
                                                                 axis=1)
    doctor_paper_data['total_paper'] = doctor_paper_data.apply(
        lambda x: (x['paper_impact'] * w1 + x['cosine_simil'] * w2) / (w1 + w2), axis=1)

    ranking = doctor_paper_data.sort_values(by='total_paper', ascending=False)
    return ranking[0:5]



input_text = ' '

while(input_text != 'exit'):

    print('\n검색하고자 하는 질병명들을 입력해주세요')
    input_text = input()

    print('\n가중치 비율을 입력하세요 ( 논문 impact : 질병 유사도 )')
    weight_paper_impact = int(input('논문 impact 가중치 : '))
    weight_sim = int(input('질병 유사도 가중치 : '))
    print('논문 impact : 질병 유사도 가중치 = ' + str(weight_paper_impact) + ' : ' + str(weight_sim))

    count = 1

    recom_list = paper_score(input_text, weight_paper_impact, weight_sim)

    print("---------------------------------------")

    for index in range(0,5):
      i = recom_list.iloc[index]
      print(str(count) + '순위')
      for key, value in i.items():
        print(key +' : '+ str(value))

      print('---------------------------------------')

      count += 1

    print('---------------------------------------')
