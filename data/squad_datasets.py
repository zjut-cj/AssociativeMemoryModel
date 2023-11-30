import json
import string
import numpy as np
import torch

from tqdm import tqdm
from typing import Tuple, List
from text2vec import SentenceModel
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader


class SquadDatasets(Dataset):
    def __init__(self):
        pass

    @staticmethod
    def load_json(path):
        '''
        Loads the JSON file of the Squad dataset.
        Returns the json object of the dataset.
        '''
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("Length of data: ", len(data['data']))
        print("Data Keys: ", data['data'][0].keys())
        print("Title: ", data['data'][0]['title'])

        return data

    @staticmethod
    def parse_data(data: dict) -> list:
        '''
        Parses the JSON file of Squad dataset by looping through the
        keys and values and returns a list of dictionaries with
        context, query and label triplets being the keys of each dict.
        '''
        data = data['data']
        qa_list = []

        for paragraphs in data:

            for para in paragraphs['paragraphs']:
                context = para['context']

                for qa in para['qas']:

                    question = qa['question']

                    for ans in qa['answers']:
                        answer = ans['text']

                        qa_dict = {}
                        qa_dict['context'] = context
                        qa_dict['question'] = question

                        qa_dict['answer'] = answer
                        qa_list.append(qa_dict)

        return qa_list

    @staticmethod
    def transform_data(data_list):
        transformed_data = []

        for item in data_list:
            context = [item["context"]]
            questions = [item["question"]]
            answers = [item["answer"]]

            context_question_answer_triplet = tuple([context, questions, answers])
            transformed_data.append(context_question_answer_triplet)

        return transformed_data


    @staticmethod
    def split_sentences(triplet):
        split_context = triplet[0][0].split()
        split_questions = triplet[1][0].split()
        split_answers = triplet[2][0].split()
        return (split_context, split_questions, split_answers)

    @staticmethod
    def clean_word(word):
        # 清洗单词，将其转换为小写，去除标点符号
        cleaned_word = word.lower().translate(str.maketrans('', '', string.punctuation))
        return cleaned_word

    @staticmethod
    def clean_triplet(triplet):
        # 清洗context
        cleaned_context = [SquadDatasets.clean_word(word) for word in triplet[0]]

        # 清洗questions
        cleaned_questions = [SquadDatasets.clean_word(word) for word in triplet[1]]

        # 清洗answers
        cleaned_answers = [SquadDatasets.clean_word(word) for word in triplet[2]]

        return (cleaned_context, cleaned_questions, cleaned_answers)

    @staticmethod
    def filter_triplet(triplet):
        # 获取第二个元素的list和第三个元素的list
        questions = triplet[1]
        answers = triplet[2]

        # 创建一个新的三元组，用于存储筛选后的数据
        filtered_triplet = list(triplet)

        # 筛选过程
        indices_to_remove = []  # 存储需要删除的索引

        for i, (question, answer) in enumerate(zip(questions, answers)):
            if len(question) > 20 or len(answer) > 10:
                indices_to_remove.append(i)

        # 从后往前删除，以防删除元素后索引变化
        indices_to_remove.reverse()
        for i in indices_to_remove:
            # 删除第二个元素中的子列表
            del questions[i]
            # 删除第三个元素中的对应子列表
            del answers[i]

        # 更新第一个元素，如果需要
        if len(questions) != len(filtered_triplet[1]):
            context = filtered_triplet[0]
            new_context = np.delete(context, indices_to_remove, axis=0)
            filtered_triplet[0] = new_context

        return tuple(filtered_triplet)

    @staticmethod
    def build_vocab(stories: List) -> Tuple[List, int]:
        # 生成新的三元组
        new_stories = (stories[0], stories[1], stories[2])

        # 初始化一个有序字典，用于存储词汇表中的单词
        vocab_dict = OrderedDict()

        # 遍历stories中的每个元素
        for s in new_stories:
            # 将问题中的单词加入有序字典
            for word in s:
                vocab_dict[word] = None  # 使用None作为占位符

        # 从有序字典中提取词汇表
        vocab = list(vocab_dict.keys())
        vocab.insert(0, 'NIL')
        vocab_size = len(vocab)

        return vocab, vocab_size

    @staticmethod
    def one_hot_encode(data, word2index):
        # 获取词典的大小
        vocab_size = len(word2index)

        # 初始化一个全零的one-hot编码矩阵
        one_hot_matrix = []

        # 对于数据中的每个单词，将其对应的索引位置设置为1
        for word in data:
            if word in word2index:
                index = word2index[word]
                one_hot_vector = [0] * vocab_size
                one_hot_vector[index] = 1
                one_hot_matrix.append(one_hot_vector)

        return one_hot_matrix

    @staticmethod
    def transform_lists_to_arrays(encoding_data):
        # 解包元组
        first_element, list_of_2d_arrays, list_of_1d_arrays = encoding_data

        # 转换第二个元素为多维数组，假设有m个二维数组，每个形状为[20, 100]
        m = len(list_of_2d_arrays)
        array_2d = np.array(list_of_2d_arrays)

        # 转换第三个元素为多维数组，假设有n个一维数组，每个形状为[10, 100]
        n = len(list_of_1d_arrays)
        array_1d = np.array(list_of_1d_arrays)

        # 创建包含原始第一个元素和转换后两个数组的新元组
        new_encoding_data = (first_element, array_2d, array_1d)

        return new_encoding_data

    # @staticmethod
    # def encoded_data(cleaned_triplet, word2index):
    #     # 调用one_hot_encode函数对每个部分进行one-hot编码
    #     encoded_data = (np.array([SquadDatasets.one_hot_encode(cleaned_triplet[0], word2index)], dtype=np.float32),
    #                     np.array([np.array(SquadDatasets.one_hot_encode(question, word2index), dtype=np.float32)
    #                               for question in cleaned_triplet[1]]),
    #                     np.array([np.array(SquadDatasets.one_hot_encode(answer, word2index), dtype=np.float32)
    #                               for answer in cleaned_triplet[2]]))
    #
    #     # 填充第一个元素的第二维度为200
    #     max_seq_length = 200
    #     for i in range(len(encoded_data[0])):
    #         padded_sequence = np.pad(encoded_data[0][i], ((0, max_seq_length - len(encoded_data[0][i])), (0, 0)),
    #                                  'constant')
    #
    #     encoded_data = (padded_sequence,) + encoded_data[1:]
    #
    #     # 填充第二个元素的每个数组的第二维度为20
    #     max_seq_length = 20
    #     padded_questions = [np.pad(question, ((0, max_seq_length - question.shape[0]), (0, 0)), 'constant') for question
    #                         in
    #                         encoded_data[1]]
    #     encoded_data = (encoded_data[0], padded_questions) + encoded_data[2:]
    #
    #     # 填充第三个元素的每个数组的第二维度为10
    #     max_seq_length = 10
    #     padded_answers = [np.pad(answer, ((0, max_seq_length - answer.shape[0]), (0, 0)), 'constant') for answer in
    #                       encoded_data[2]]
    #     encoded_data = encoded_data[:2] + (padded_answers,)
    #
    #     return SquadDatasets.transform_lists_to_arrays(encoded_data)

    # @staticmethod
    # def encoded_data(cleaned_triplet, word2index):
    #     # 对第一个元素编码
    #     context_encoding = [word2index[word] if word in word2index else 0 for word in cleaned_triplet[0]]
    #     # 填充到长度200
    #     context_encoding += [0] * (200 - len(context_encoding))
    #     context_encoding = np.array(context_encoding, dtype=np.int64)
    #
    #     # 对第二个元素中的列表编码
    #     question_encodings = []
    #     for question in cleaned_triplet[1]:
    #         question_encoding = [word2index[word] if word in word2index else 0 for word in question]
    #         # 填充到长度20
    #         question_encoding += [0] * (20 - len(question_encoding))
    #         question_encodings.append(np.array(question_encoding, dtype=np.int64))
    #
    #     # 对第三个元素中的列表编码
    #     answer_encodings = []
    #     for answer in cleaned_triplet[2]:
    #         answer_encoding = [word2index[word] if word in word2index else 0 for word in answer]
    #         # 填充到长度10
    #         answer_encoding += [0] * (10 - len(answer_encoding))
    #         answer_encodings.append(np.array(answer_encoding, dtype=np.int64))
    #
    #     return SquadDatasets.transform_lists_to_arrays((context_encoding, question_encodings, answer_encodings))

    # @staticmethod
    # def encoded_data(cleaned_triplet, word2index):
    #     # 对第一个元素编码并调整形状为 [1, 1, 200]
    #     context_encoding = [word2index[word] / len(word2index) if word in word2index else 0.0 for word in
    #                         cleaned_triplet[0]]
    #     context_encoding += [0.0] * (200 - len(context_encoding))
    #     context_encoding = np.array(context_encoding, dtype=np.float32).reshape(1, 1, 200)
    #
    #     # 对第二个元素中的列表编码并调整形状为 [n, 1, 200]
    #     question_encoding = [word2index[word] / len(word2index) if word in word2index else 0.0 for word in
    #                           cleaned_triplet[1]]
    #     question_encoding += [0.0] * (200 - len(question_encoding))
    #     question_encoding = np.array(question_encoding, dtype=np.float32).reshape(1, 1, 200)
    #
    #     # 对第三个元素中的列表编码并调整形状为 [n, 1, 10]
    #     answer_encoding = [word2index[word] if word in word2index else 0.0 for word in
    #                         cleaned_triplet[2]]
    #     answer_encoding += [0.0] * (100 - len(answer_encoding))
    #     answer_encoding = np.array(answer_encoding, dtype=np.float32)
    #     return SquadDatasets.transform_lists_to_arrays((context_encoding, question_encoding, answer_encoding))

    @staticmethod
    def encoded_data(cleaned_triplet, word2index):
        # 调用one_hot_encode函数对每个部分进行one-hot编码
        encoded_data = (np.array([SquadDatasets.one_hot_encode(cleaned_triplet[0], word2index)], dtype=np.float32),
                        np.array([SquadDatasets.one_hot_encode(cleaned_triplet[1], word2index)], dtype=np.float32))

        # 填充第一个元素的第二维度为200
        max_seq_length = 200
        for i in range(len(encoded_data[0])):
            padded_sequence = np.pad(encoded_data[0][i], ((0, max_seq_length - len(encoded_data[0][i])), (0, 0)),
                                     'constant')

        encoded_data = (padded_sequence,) + encoded_data[1:]

        # 填充第二个元素的每个数组的第二维度为20
        max_seq_length = 200
        for i in range(len(encoded_data[1])):
            padded_questions = np.pad(encoded_data[1][i], ((0, max_seq_length - len(encoded_data[1][i])), (0, 0)),
                                      'constant')
        encoded_data = (encoded_data[0], padded_questions) + encoded_data[2:]

        # answer_encoding = [word2index[word] if word in word2index else 0.0 for word in
        #                         cleaned_triplet[2]]
        # answer_encoding += [0.0] * (10 - len(answer_encoding))
        # answer_encoding = np.array(answer_encoding, dtype=np.float32)
        # encoded_data = encoded_data[:2] + (answer_encoding,)
        answer_encoding = np.zeros(len(word2index), dtype=np.float32)
        # 遍历 word_list 中的每个单词
        for word in cleaned_triplet[2]:
            # 查找单词在词典中的索引
            index = word2index.get(word, -1)

            # 如果找到了，设置对应位置为 1
            if index != -1:
                answer_encoding[index] = 1

        encoded_data = encoded_data[:2] + (answer_encoding,)

        return encoded_data

    @staticmethod
    def normalization_embedding(cleaned_triplet, word2index):
        wordsList = np.load('/home/jww/storage/glove.6B/wordsList_300d.npy')

        wordsList = wordsList.tolist()  # Originally loaded as numpy array
        wordVectors = np.load('/home/jww/storage/glove.6B/wordVectors_300d.npy')

        mean_value = np.mean(wordVectors)
        variance_value = np.var(wordVectors)
        left_boundary = mean_value - 3 * np.sqrt(variance_value)
        right_boundary = mean_value + 3 * np.sqrt(variance_value)
        zero_embedding = np.array([0] * 300, dtype=np.float32)

        context, questions, answers = cleaned_triplet
        context_encoded = []
        for word in context:
            try:
                word_index = wordsList.index(word)
                word_embedding = wordVectors[word_index]
            except ValueError:
                word_embedding = zero_embedding
            embedding_n01 = (word_embedding - np.array([mean_value] * 300)) / np.array([np.sqrt(variance_value)] * 300)
            embedding_norm = np.array([0] * 300, dtype=np.float32)
            embedding_norm = np.array([0] * 300, dtype=np.float32)
            for k in range(300):
                if word_embedding[k] < left_boundary:
                    embedding_norm[k] = -3
                elif word_embedding[k] > right_boundary:
                    embedding_norm[k] = 3
                else:
                    embedding_norm[k] = embedding_n01[k]
            embedding_norm = (embedding_norm + np.array([np.abs(3)] * 300)) / (3 * 2)
            embedding_norm = np.clip(embedding_norm, a_min=0, a_max=1)
            context_encoded.append(embedding_norm)
        for i in range(150 - len(context_encoded)):
            context_encoded.append(zero_embedding)

        questions_encoded = []
        for word in questions:
            try:
                word_index = wordsList.index(word)
                word_embedding = wordVectors[word_index]
            except ValueError:
                word_embedding = zero_embedding
            embedding_n01 = (word_embedding - np.array([mean_value] * 300)) / np.array([np.sqrt(variance_value)] * 300)
            embedding_norm = np.array([0] * 300, dtype=np.float32)
            embedding_norm = np.array([0] * 300, dtype=np.float32)
            for k in range(300):
                if word_embedding[k] < left_boundary:
                    embedding_norm[k] = -3
                elif word_embedding[k] > right_boundary:
                    embedding_norm[k] = 3
                else:
                    embedding_norm[k] = embedding_n01[k]
            embedding_norm = (embedding_norm + np.array([np.abs(3)] * 300)) / (3 * 2)
            embedding_norm = np.clip(embedding_norm, a_min=0, a_max=1)
            questions_encoded.append(embedding_norm)
        for i in range(20 - len(questions_encoded)):
            questions_encoded.append(zero_embedding)

        encoded_data = (np.array(context_encoded, dtype=np.float32), np.array(questions_encoded, dtype=np.float32))

        answer_encoding = np.zeros(len(word2index), dtype=np.float32)
        # 遍历 word_list 中的每个单词
        for word in answers:
            # 查找单词在词典中的索引
            index = word2index.get(word, -1)

            # 如果找到了，设置对应位置为 1
            if index != -1:
                answer_encoding[index] = 1

        encoded_data = encoded_data[:2] + (answer_encoding,)

        return encoded_data

    # def process_data(self, train_file_path, valid_file_path):
    #     # 加载Json数据S
    #     train_data = self.load_json(train_file_path)
    #     valid_data = self.load_json(valid_file_path)
    #
    #     # 解析Json数据
    #     train_list = self.parse_data(train_data)
    #     valid_list = self.parse_data(valid_data)
    #
    #     # 转换train_list和valid_list
    #     transformed_train_data = self.transform_data(train_list)
    #     transformed_valid_data = self.transform_data(valid_list)
    #
    #     train_set = []
    #     print("Loading train sets...")
    #     for i in tqdm(range(len(transformed_train_data))):
    #         data = transformed_train_data[i]
    #         split_data = self.split_sentences(data)
    #         if len(split_data[0]) > 200:
    #             continue
    #         cleaned_data = self.clean_triplet(split_data)
    #         if len(cleaned_data[1]) > 20 or len(cleaned_data[2]) > 10:
    #             continue
    #         vocab, vocab_size = self.build_vocab(cleaned_data)
    #         if vocab_size > 100:
    #             continue
    #         word2index = {w: i for i, w in enumerate(vocab)}
    #         # while len(word2index) < 100:
    #         #     word2index['NIL' + str(len(word2index))] = len(word2index)
    #         encoding_data = self.encoded_data(cleaned_data, word2index)
    #         train_set.append(encoding_data)
    #
    #     test_set = []
    #     print("Loading test sets...")
    #     for i in tqdm(range(len(transformed_valid_data))):
    #         data = transformed_train_data[i]
    #         split_data = self.split_sentences(data)
    #         if len(split_data[0]) > 200:
    #             continue
    #         cleaned_data = self.clean_triplet(split_data)
    #         if len(cleaned_data[1]) > 20 or len(cleaned_data[2]) > 10:
    #             continue
    #         vocab, vocab_size = self.build_vocab(cleaned_data)
    #         if vocab_size > 100:
    #             continue
    #         word2index = {w: i for i, w in enumerate(vocab)}
    #         # while len(word2index) < 100:
    #         #     word2index['NIL' + str(len(word2index))] = len(word2index)
    #         encoding_data = self.encoded_data(cleaned_data, word2index)
    #         # if len(encoding_data[1] == 0) or len(encoding_data[2] == 0):
    #         #     continue
    #         test_set.append(encoding_data)
    #
    #     return train_set, test_set

    def process_data(self, train_file_path, valid_file_path, train_qas_pairs, test_qas_pairs):
        # 加载Json数据S
        train_data = self.load_json(train_file_path)
        valid_data = self.load_json(valid_file_path)

        # 解析Json数据
        train_list = self.parse_data(train_data)
        valid_list = self.parse_data(valid_data)

        # 转换train_list和valid_list
        transformed_train_data = self.transform_data(train_list)
        transformed_valid_data = self.transform_data(valid_list)

        train_set = []
        train_paragraph_set = []
        train_split_data_set = []
        train_cleaned_data_set = []
        train_word2index_set = []
        train_vocab_size = []
        print("Loading train sets...")
        train_data_size = 0
        for i in tqdm(range(len(transformed_train_data))):
            data = transformed_train_data[i]
            split_data = self.split_sentences(data)
            if len(split_data[0]) > 150:
                continue
            cleaned_data = self.clean_triplet(split_data)
            if len(cleaned_data[1]) > 20:
                continue
            vocab, vocab_size = self.build_vocab(cleaned_data)
            if vocab_size > 100:
                continue
            word2index = {w: i for i, w in enumerate(vocab)}
            while len(word2index) < 100:
                word2index['NIL' + str(len(word2index))] = len(word2index)
            # encoding_data = self.encoded_data(cleaned_data, word2index)
            encoding_data = self.normalization_embedding(cleaned_data, word2index)
            train_paragraph_set.append(data)
            train_split_data_set.append(split_data)
            train_cleaned_data_set.append(cleaned_data)
            train_word2index_set.append(word2index)
            train_set.append(encoding_data)
            train_vocab_size.append(vocab_size)
            train_data_size = train_data_size + 1
            if train_data_size == train_qas_pairs:
                break

        test_set = []
        test_paragraph_set = []
        test_split_data_set = []
        test_cleaned_data_set = []
        test_word2index_set = []
        test_vocab_size = []
        print("Loading test sets...")
        test_data_size = 0
        for i in tqdm(range(len(transformed_valid_data))):
            data = transformed_train_data[i]
            split_data = self.split_sentences(data)
            if len(split_data[0]) > 150:
                continue
            cleaned_data = self.clean_triplet(split_data)
            if len(cleaned_data[1]) > 20:
                continue
            vocab, vocab_size = self.build_vocab(cleaned_data)
            if vocab_size > 100:
                continue
            word2index = {w: i for i, w in enumerate(vocab)}
            while len(word2index) < 100:
                word2index['NIL' + str(len(word2index))] = len(word2index)
            # encoding_data = self.encoded_data(cleaned_data, word2index)
            encoding_data = self.normalization_embedding(cleaned_data, word2index)
            # if len(encoding_data[1] == 0) or len(encoding_data[2] == 0):
            #     continue
            test_paragraph_set.append(data)
            test_split_data_set.append(split_data)
            test_cleaned_data_set.append(cleaned_data)
            test_word2index_set.append(word2index)
            test_vocab_size.append(vocab_size)
            test_set.append(encoding_data)
            test_data_size = test_data_size + 1
            if test_data_size == test_qas_pairs:
                break

        return (train_set, test_set), (train_paragraph_set, test_paragraph_set), \
               (train_split_data_set, test_split_data_set), (train_cleaned_data_set, test_cleaned_data_set), \
               (train_word2index_set, test_word2index_set), (train_vocab_size, test_vocab_size)


if __name__ == "__main__":
    train_file_path = '/home/jww/storage/squad/train-v1.1.json'
    valid_file_path = '/home/jww/storage/squad/dev-v1.1.json'
    squad_datasets = SquadDatasets()
    data_set, paragraph_set, split_data_set, cleaned_data_set, word2index_set = \
        squad_datasets.process_data(train_file_path=train_file_path, valid_file_path=valid_file_path)

    train_set, test_set = data_set
    print(len(train_set))
    print(len(test_set))

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print(len(train_loader))
    print(len(test_loader))

    for i, sample in enumerate(train_loader):
        context, questions, answers = sample
        print("context: ", context)
        print("--------------------------------")
        for j in range(len(questions)):
            print("Question{j}: ", questions[j])
        print("--------------------------------")
        for k in range(len(answers)):
            print("Answer{k}: ", answers[k])
        break

