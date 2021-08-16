import pandas as pd
import pm4py
import gensim
import nltk
import lxml
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler

import os
import sys

from datetime import datetime

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from glove import Corpus, Glove

# from representations.Act2Vec import learn as act2vec_learn
# from representations.Trace2Vec import learn as trace2vec_learn
# from representations.loadXES import get_sentences_XES

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.util import dataframe_utils


def get_day_part(hour):
    period = (hour % 24 + 4) // 4

    part_by_hour = {
        1: 'Late Night',
        2: 'Early Morning',
        3: 'Morning',
        4: 'Noon',
        5: 'Evening',
        6: 'Night'
    }

    return part_by_hour[period]


def get_time_attr(column, attr, time_format):
    return getattr(column.map(lambda cur_time: datetime.strptime(cur_time, time_format)).dt, attr)


def act2vec_learn(sentences, vectorsize, name='traces'):
    model = gensim.models.Word2Vec(sentences, vector_size=vectorsize, window=3, min_count=0)
    nrEpochs= 10
    for epoch in range(nrEpochs):
        if epoch % 2 == 0:
            print ('Now training epoch %s'%epoch)
        model.train(sentences, start_alpha=0.025, epochs=nrEpochs, total_examples=model.corpus_count)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    model.save(f'output/act2vec_{name}_A2VVS'+str(vectorsize) +'.model')

    return model


def glove_learn(sentences, vectorsize, name='traces'):
    corpus_model = Corpus()
    corpus_model.fit(sentences, window=10)
    glove = Glove(no_components=16, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=10,
                  no_threads=True, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)

    glove.save(f'output/act2vec_{name}_GLOVE' + str(vectorsize) + '.model')

    return glove


def trace2vec_learn(documents, vectorsize, name='trace2vec'):
    print (str(len(documents)), 'lines found.')

    model = gensim.models.Doc2Vec(documents, dm = 0, alpha=0.025, vector_size= vectorsize, window=3, min_alpha=0.025, min_count=0)

    nrEpochs= 10
    for epoch in range(nrEpochs):
        if epoch % 2 == 0:
            print ('Now training epoch %s'%epoch)
        model.train(documents, total_examples=len(documents), epochs=nrEpochs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    model.save('output/' + name + 'T2VVS'+str(vectorsize) +'.model')
    model.save_word2vec_format('output/' + name + 'T2VVS'+str(vectorsize) + '.word2vec')

    return model


def cut_and_extract_features(path_to_dataset='data/helpdesk.csv', name='helpdesk', time_format='%Y/%m/%d %H:%M:%S.%f', output_suffix='', vectorize=16, use_glove=False):
    print('Loading data...')
    log_data = pd.read_csv(path_to_dataset)

    if 'bpi' in name or '_lite' in name:
        try:
            log_data.drop(columns=['startTime'], inplace=True)
        except Exception:
            pass
        log_data.columns = ['Case ID', 'Activity', 'Complete Timestamp']
    elif 'helpdesk' in name:
        pass
        """
        n_rows_prev = log_data.shape[0]
        log_data = log_data[['Case ID', 'Activity', 'Complete Timestamp']]

        case_ids = log_data['Case ID'].unique()
        for case_id in case_ids:
            if not log_data[
                (log_data['Case ID'] == case_id) &
                (log_data['Activity'] == 'Take in charge ticket')
            ].shape[0] or not log_data[
                (log_data['Case ID'] == case_id) &
                (log_data['Activity'] == 'Assign seriousness')
            ].shape[0]:
                print(f'drop case_id = {case_id}')
                log_data = log_data[log_data['Case ID'] != case_id]
        log_data.reset_index(inplace=True)
        print(f'dropped {n_rows_prev - log_data.shape[0]} rows -> {(n_rows_prev - log_data.shape[0]) / n_rows_prev}')
        """
    get_timestamp = lambda cur_time: datetime.strptime(cur_time,
                           time_format).timestamp()

    """
    print('Drop tails...')
    activity_counts = dict(log_data['Activity'].value_counts())
    activity_sum = sum(activity_counts.values())

    n_rows_prev = log_data.shape[0]
    cases_to_drop = set()
    case_ids = log_data['Case ID'].unique()
    for case_id in case_ids:
        case_data = log_data[log_data['Case ID'] == case_id]
        for act in case_data['Activity'].unique():
            if activity_counts[act] / activity_sum < 0.01:
                cases_to_drop.add(case_id)
                print(f'drop case_id = {case_id}')
                break

    log_data = log_data[~log_data['Case ID'].isin(cases_to_drop)].reset_index(drop=True)

    # log_data.reset_index(inplace=True)
    print(f'dropped {n_rows_prev - log_data.shape[0]} rows -> {(n_rows_prev - log_data.shape[0]) / n_rows_prev}')
    """

    print(log_data['Activity'].value_counts())

    mapping = {}
    for i, act in enumerate(log_data['Activity'].unique()):
        mapping[act] = i

    log_data['Activity ID'] = log_data['Activity'].map(lambda x: mapping[x])

    print('Day part...')
    log_data['Day part'] = log_data['Complete Timestamp'].map(lambda cur_time: datetime.strptime(cur_time, time_format)).dt.hour.map(lambda x: get_day_part(x))

    print('Timestamp attrs...')

    time_attrs = ['hour', 'dayofweek', 'month', 'year']

    for attr in time_attrs:
        log_data[attr] = get_time_attr(log_data['Complete Timestamp'], attr, time_format)

    log_data['Timestamp'] = log_data['Complete Timestamp'].map(lambda x: get_timestamp(x)).astype('float64')
    # log_data = log_data.sort_values('Timestamp')
    log_data.drop(columns=['Complete Timestamp'], inplace=True)
    # renamed_data = log_data.rename(columns={'Activity' : "concept:name", 'Case ID' : "case:CaseID"})
    # parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:CaseID'}
    # event_log = log_converter.apply(renamed_data, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
    print(f'UNIQUE DAY PARTS: {log_data["Day part"].value_counts()}')
    log_data['static_index'] = pd.Series(np.arange(log_data.shape[0]))

    print('Previous activity...')
    for case_id in log_data['Case ID'].unique():
        case_data = log_data[log_data['Case ID'] == case_id].sort_values('Timestamp').reset_index(drop=True)
        for cnt, idx in enumerate(case_data['static_index']):
            if cnt == 0:
                log_data.loc[log_data.static_index == idx, 'Previous Activity'] = 'None'
            else:
                log_data.loc[log_data.static_index == idx, 'Previous Activity'] = log_data.loc[log_data.static_index == idx - 1, 'Activity'].values[0]

    """
    print('Activity number in case...')
    for case_id in log_data['Case ID'].unique():
        case_data = log_data[log_data['Case ID'] == case_id].sort_values('Timestamp').reset_index()
        log_data.loc[log_data['Case ID'] == case_id, 'Activity order number'] = np.arange(case_data.shape[0])
    """

    print('Timestamp delta: with previous activity in case...')
    for case_id in log_data['Case ID'].unique():
        case_data = log_data[log_data['Case ID'] == case_id].sort_values('Timestamp').reset_index(drop=True)
        previous = pd.concat([pd.Series([case_data.Timestamp[0]]), case_data.Timestamp[:-1]], ignore_index=True)
        delta = case_data.Timestamp - previous
        log_data.loc[log_data['Case ID'] == case_id, 'Delta from prev'] = np.array(delta)

    print('Timestamp delta: with start activity in case...')
    for case_id in log_data['Case ID'].unique():
        case_data = log_data[log_data['Case ID'] == case_id].sort_values('Timestamp').reset_index(drop=True)
        start = case_data.Timestamp[0]
        delta = case_data.Timestamp - start
        log_data.loc[log_data['Case ID'] == case_id, 'Delta from start'] = np.array(delta)

    print('W/resp to  average...')
    total_count = 0
    for case_id in log_data['Case ID'].unique():
        total_count += (log_data['Case ID'] == case_id).sum()
    average_count = total_count / log_data['Case ID'].unique().shape[0]
    for case_id in log_data['Case ID'].unique():
        case_data = log_data[log_data['Case ID'] == case_id].sort_values('Timestamp')
        log_data.loc[log_data['Case ID'] == case_id, 'To Average'] = np.arange(1, case_data.shape[0] +
        1) / average_count

    print('Embedding...')

    sentences = []
    log_data['Activity extended'] = log_data['Activity'].astype(str) + '_' + log_data['Day part'].astype(str)
    for case_id in log_data['Case ID'].unique():
        case_data = log_data[log_data['Case ID'] == case_id].sort_values('Timestamp')
        to_embed = case_data['Activity'].astype(str)
        sentences.append(list(to_embed.to_list()))

    tagged_documents = []
    for case_id in log_data['Case ID'].unique():
        case_data = log_data[log_data['Case ID'] == case_id].sort_values('Timestamp').reset_index(drop=True)
        sentence = case_data['Activity'].astype(str).to_list()
        td = TaggedDocument(sentence, [case_id])
        tagged_documents.append(td)

    vectoring_func = [act2vec_learn, glove_learn]

    model = vectoring_func[use_glove](sentences, vectorize, name)

    vectors = []
    for i in range(log_data.shape[0]):
        if not use_glove:
            vec = model.wv[str(log_data.loc[i, 'Activity'])]
        else:
            word_idx = model.dictionary[str(log_data.loc[i, 'Activity'])]
            vec = model.word_vectors[word_idx]
        vectors.append(vec)
    vectors = np.array(vectors)

    doc2vec = trace2vec_learn(tagged_documents, vectorize)

    to_append = pd.DataFrame(vectors, columns=[f'vec_{i}' for i in range(vectorize)])
    log_data = pd.concat([log_data, to_append], axis=1)

    for case_id in log_data['Case ID'].unique():
        case_data = log_data[log_data['Case ID'] == case_id].sort_values('Timestamp')
        prefix = []
        vectors = []
        for act in case_data['Activity'].astype(str):
            prefix.append(act)
            vect = doc2vec.infer_vector(prefix)
            assert len(vect) == vectorize
            vectors.append(vect)
        vectors = np.array(vectors)

        log_data.loc[log_data['Case ID'] == case_id, [f'trace2vec_{i}' for i in range(vectorize)]] = vectors
    print('One hot...')
    log_data = pd.get_dummies(log_data, columns=["Activity", "Activity extended", "Previous Activity", "Day part"] + time_attrs, prefix=["activity", "activity_ext", "prev_activity", "day_part"] + time_attrs)

    print('Scale...')
    numeric = [
        # 'Timestamp',
        'Delta from prev',
        'Delta from start',
        'To Average'
    ]

    log_data.loc[:, numeric] = StandardScaler().fit_transform(log_data.loc[:, numeric])

    log_data.drop(columns=['static_index'], inplace=True)

    print('Saving...')

    # print(log_data.columns)

    # print(log_data[[f'trace2vec_{i}' for i in range(vectorize)]].head(15))
    print(log_data[[f'vec_{i}' for i in range(vectorize)]].head(15))

    if not os.path.exists('preprocessed'):
        os.makedirs('preprocessed')

    if 'index' in log_data.columns:
        log_data.drop(columns=['index'], inplace=True)

    log_data.to_csv(f'preprocessed/{name}_formatted{"_" * bool(len(output_suffix))}{output_suffix}.csv')

    return log_data

cut_and_extract_features(path_to_dataset=sys.argv[1], name=sys.argv[2], time_format=sys.argv[3], output_suffix=sys.argv[4], vectorize=16, use_glove=False)
