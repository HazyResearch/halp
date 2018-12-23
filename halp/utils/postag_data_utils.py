import nltk
from nltk.stem import PorterStemmer
import numpy as np
import sys, os
import torch
from halp.utils.utils import set_seed
from torch.utils.data.dataset import Dataset
import _pickle as cp
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('')

DOWNLOAD_PATH = "./datasets/nltk/"
nltk.data.path.append(DOWNLOAD_PATH)


def download_postag_dataset(download_path):
    nltk.download('conll2000', download_dir=DOWNLOAD_PATH)
    nltk.download('treebank', download_dir=DOWNLOAD_PATH)
    nltk.download('universal_tagset', download_dir=DOWNLOAD_PATH)


# def process_data(dataset="conll2000"):
#     set_seed(0)
#     ps = PorterStemmer()
#     if dataset == "conll2000":
#         logger.info("Start processing " + dataset)
#         train_sentences = nltk.corpus.conll2000.tagged_sents(
#             "train.txt", tagset='universal')
#         test_sentences = nltk.corpus.conll2000.tagged_sents(
#             "test.txt", tagset='universal')
#         train_tags = set(
#             [tag for sentence in train_sentences for _, tag in sentence])

#         train_words = set(
#             [ps.stem(w) for sentence in train_sentences for w, _ in sentence])

#         # test_all_words = [
#         #     ps.stem(w) for sentence in test_sentences
#         #     for w, _ in sentence
#         # ]

#         test_tags = set(
#             [tag for sentence in test_sentences for _, tag in sentence])

#         test_words = set(
#             [ps.stem(w) for sentence in test_sentences for w, _ in sentence])
#         raise Exception("conll2000 dataset is not supported yet.")

#     elif dataset == "treebank":
#         logger.info("Start processing " + dataset)
#         sentences = nltk.corpus.treebank.tagged_sents(tagset='universal')

#         def get_dict(self, sentences):
#             tags = set([tag for sentence in sentences for _, tag in sentence])

#             words = set(
#                 [ps.stem(w) for sentence in sentences for w, _ in sentence])
#             tag_dict = {}
#             word_dict = {}
#             assert type(tags) == set
#             assert type(words) == set
#             for i, tag in enumerate(tags):
#                 tag_dict[tag] == i
#             for i, word in enumerate(words):
#                 word_dict[word] == i
#             return tag_dict, word_dict

#         tag_dict, word_dict = get_dict(sentences)
#         train_ratio = 0.8
#         idx = np.random.permutation(len(sentences)).tolist()
#         split_idx = int(train_ratio * len(idx))
#         train_sentences = sentences[idx[:split_idx]]
#         test_sentences = sentences[idx[split_idx:]]



#         # length = [len(sentence) for sentence in sentences]

#         # print(np.max(length), np.mean(length), sorted(length))

#         # print("test ", len(sentences))
#         # raise Exception("Tree bank dataset support is not setup.")
#     # print(len(test_words - train_words), len(test_words), len(train_words), len(test_tags), len(train_tags))
#     # cnt = 0
#     # for x in test_all_words:
#     #     if x not in train_words:
#     #         cnt += 1
#     # print("unk works ", cnt, len(test_all_words))
#     # assert test_words < train_words
#     # assert test_tags < train_tags

def process_data(data_path=DOWNLOAD_PATH + "/treebank/processed/"):
    os.makedirs(data_path, exist_ok=True)
    set_seed(0)
    ps = PorterStemmer()
    logger.info("Start processing dataset")
    sentences = nltk.corpus.treebank.tagged_sents(tagset='universal')

    def get_dict(sentences):
        tags = set([tag for sentence in sentences for _, tag in sentence])

        words = set(
            [ps.stem(w) for sentence in sentences for w, _ in sentence])
        tag_dict = {}
        word_dict = {}
        assert type(tags) == set
        assert type(words) == set
        for i, tag in enumerate(tags):
            tag_dict[tag] = i
        for i, word in enumerate(words):
            word_dict[word] = i
        return tag_dict, word_dict

    tag_dict, word_dict = get_dict(sentences)
    train_ratio = 0.8
    idx = np.random.permutation(len(sentences)).tolist()
    split_idx = int(train_ratio * len(idx))
    train_sentences = [sentences[i] for i in idx[:split_idx]]
    test_sentences = [sentences[i] for i in idx[split_idx:]]
    train_sentences= [[(ps.stem(word[0]), word[1]) for word in sentence] for sentence in train_sentences]
    test_sentences= [[(ps.stem(word[0]), word[1]) for word in sentence] for sentence in test_sentences]

    with open(data_path + "trainset", "wb") as f:
        cp.dump(train_sentences, f)

    with open(data_path + "testset", "wb") as f:
        cp.dump(test_sentences, f)

    with open(data_path + "tag_dict", "wb") as f:
        cp.dump(tag_dict, f)

    with open(data_path + "word_dict", "wb") as f:
        cp.dump(word_dict, f)
    logger.info("Processing dataset done.")


class TaggingDataset(Dataset):
    def __init__(self, sentences, tag_dict, word_dict):
        self.tag_dict, self.word_dict = tag_dict, word_dict
        self.words = [[self.word_dict[word] for word, _ in sentence]
                      for sentence in sentences]
        self.tags = [[self.tag_dict[tag] for _, tag in sentence]
                     for sentence in sentences]
        self.length = [len(x) for x in self.words]
        self.max_seq_length = max(self.length)
        logger.info("max seq length in dataset: " + str(self.max_seq_length))
        # inflate the first example to make sure we
        # allocate enough memory for the bc layer cache
        self.words[0] = self.words[0] + [0] * (self.max_seq_length - len(self.words[0]))
        self.tags[0] = self.tags[0] + [-1] * (self.max_seq_length - len(self.tags[0]))
        assert len(self.words) == len(self.tags)
        assert len(self.words[-1]) == len(self.tags[-1])

    def __getitem__(self, index):
        return self.words[index], self.tags[index]

    def __len__(self):
        return len(self.words)


def collate_fn(data):
    words, tags = zip(*data)
    lengths = [len(x) for x in words]
    X = torch.zeros(len(words), max(lengths)).long()
    Y = -torch.ones(len(tags), max(lengths)).long()
    for i, (word_seq, tag_seq) in enumerate(zip(words, tags)):
        length = lengths[i]
        # print("check ", len(word_seq), len(tag_seq))
        X[i, :length] = torch.LongTensor(word_seq)
        Y[i, :length] = torch.LongTensor(tag_seq)
    # print("input shape ", X.shape, Y.shape)
    return X, Y

def get_treebank_data_loader(data_path=DOWNLOAD_PATH + "/treebank/processed/", args=None):
    with open(data_path + "trainset", "rb") as f:
        train_sentences = cp.load(f)
    with open(data_path + "testset", "rb") as f:
        test_sentences = cp.load(f)
    with open(data_path + "tag_dict", "rb") as f:
        tag_dict = cp.load(f)
    with open(data_path + "word_dict", "rb") as f:
        word_dict = cp.load(f)
    max_seq_length = 271
    num_embeddings = len(word_dict)
    # train_sentences = cp.load(data_path + "/trainset")
    # test_sentences = cp.load(data_path + "/testset")
    # tag_dict = cp.load(data_path + "/tag_dict")
    # word_dict = cp.load(data_path + "/word_dict")

    # train_sentences = train_sentences[:128]

    train_set = TaggingDataset(train_sentences, tag_dict, word_dict)
    test_set = TaggingDataset(test_sentences, tag_dict, word_dict)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    # print("check data length ", len(train_set))
    return train_data_loader, test_data_loader, None, len(train_set), max_seq_length, num_embeddings


if __name__ == "__main__":
    download_postag_dataset(DOWNLOAD_PATH)
    process_data()
