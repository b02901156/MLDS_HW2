# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:11:44 2018

@author: zhewei
"""
import os
import pickle
from collections import Counter

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
    
class file_preprocessor(object):
    def __init__(self):
        self.data_path = 'data'
        self.corpus_name = 'clr_conversation.txt'
        self.preprocessed_path = 'processed_data'
        self.vocab_size = 3000
        
        self.questions = []
        self.answers = []
        
        self.train_enc = []
        self.train_dec = []
        
        self.train_ids_enc = []
        self.train_ids_dec = []
        
        self.vocab_list = []
        self.word2id = {}
        self.id2word = {}
    
    def get_corpus(self):
        print("Reading corpus ... ")
        file_path = os.path.join(self.data_path, self.corpus_name)
        corpus = []
        with open(file_path, 'r', encoding='utf8') as fin:
            for line in fin:
                corpus.append(line.split('\n')[0])   
        return corpus

    def make_question_answer(self):
        print("Making corpus Question-Answer form... ")
              
        corpus = self.get_corpus()
        # find the separate "+++$+++" position
        sep_idx = [idx for idx, _ in enumerate(corpus) if corpus[idx]=='+++$+++' ]
        sep_idx.insert(0, int(-1))
        
        # make the Q-A corpus
        for parts in range(len(sep_idx) - 1):
            part_min = sep_idx[parts]
            part_max = sep_idx[parts + 1]
            for line_id in range(part_min + 1, part_max): 
                question_idx = line_id
                answer_idx = line_id + 1
                if answer_idx >= part_max:
                    break
                else:
                    self.questions.append(corpus[question_idx])
                    self.answers.append(corpus[answer_idx])
        for i in range(len(self.questions)):
            self.questions[i] = [word for sentence in self.questions[i] for word in sentence if word != ' ']
            self.answers[i] = [word for sentence in self.answers[i] for word in sentence if word != ' ']
            
            if(len(self.questions[i]) == 0 or len(self.answers[i]) == 0):
                continue
            self.train_enc.append(self.questions[i])
            self.train_dec.append(self.answers[i])
            
    '''        
    def prepare_dataset(self):
        print("Saving the dataset... ")
         # create path to store all the train & test encoder & decoder
        make_dir(self.preprocessed_path)
        
        # write to outfile
        filenames = ['train.enc', 'train.dec']
        files = []
        for filename in filenames:
            files.append(open(os.path.join(self.preprocessed_path, filename),'w', encoding='utf8'))
    
        for i in range(len(self.questions)):
            self.train_enc.append(self.questions[i])
            self.train_dec.append(self.answers[i])
            files[0].write(self.questions[i] + ' \n')
            files[1].write(self.answers[i] + ' \n')
    
        for file in files:
            file.close()
    '''
    def make_dict(self):
        print("Making dictionary ...")
        vocabs = []
        for line in self.questions:
            vocabs.extend(line)
            
        counter = Counter(vocabs).most_common(self.vocab_size)
        vocab_list = [w for w, _ in counter]
        keywords = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.vocab_list = keywords + vocab_list
        
        print("Writing to vocab ...")
        vocab_path = os.path.join(self.preprocessed_path, 'vocab')       
        with open(vocab_path, 'w', encoding='utf8') as fout:
            for line in self.vocab_list:
                fout.write(line)
                fout.write('\n')
                
    def word_to_id(self):
        print("Translating word to id ...")
        self.word2id = {k: v for v, k in list(enumerate(self.vocab_list))}
        self.id2word = {k: v for k, v in list(enumerate(self.vocab_list))}
        
        filenames = ['train.enc', 'train.dec']
        contents = [self.train_enc, self.train_dec]
        for idx in range(len(filenames)):
            content_id = []
    
            print("Translating {} file ...".format(filenames[idx]))
            for line in contents[idx]:
                tmp=[]
                for word in line:
                    if word not in self.word2id:
                        tmp.append(self.word2id['<UNK>'])
                    else:
                        tmp.append(self.word2id[word])
                content_id.append(tmp)
            
            print("Writing to {} file ...".format(filenames[idx]))
            outfile_path =  os.path.join(self.preprocessed_path, 'id_' + filenames[idx])
            with open(outfile_path, 'w', encoding='utf8') as fout:
                for line in content_id:
                    for word in line:
                        fout.write(str(word) + ' ')
                    fout.write('\n')
            print("{} completed ".format(filenames[idx]))
            
            if idx == 0: self.train_ids_enc = content_id
            elif idx == 1: self.train_ids_dec = content_id
     
    def save_file(self):
        print("Dumping trainFile ...")
        self.train_samples = [[q, a] for q, a in zip(self.train_ids_enc, self.train_ids_dec)]
        content = {'word2id': self.word2id, 
                   'id2word': self.id2word,
                   'trainSample': self.train_samples}
        
        dumpPath = os.path.join(self.preprocessed_path, 'trainFile.pkl')
        pickle.dump(content, open(dumpPath, 'wb'))
    
if __name__ == "__main__":
    fp = file_preprocessor()        
    fp.make_question_answer()
    #fp.prepare_dataset()
    fp.make_dict()
    fp.word_to_id()
    fp.save_file()

