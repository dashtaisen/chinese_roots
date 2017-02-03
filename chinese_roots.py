"""
Name: Nicholas A Miller
Email: nicholas.anthony.miller@gmail.com
Description: Chinese words and roots toolkit
"""

import nltk
import gensim
from gensim.models import Word2Vec
from nltk.corpus import sinica_treebank as sinica
from collections import defaultdict

#The following are for CC-CEDICT:
import csv, re

class SinicaPOS:
    """
    Class for analyzing compound words in Chinese tagged corpus. Uses traditional characters (繁體字)
    
    Required for init: treebank corpus or other SyntaxCorpusReader object

    Tested on NLTK's Sinica Treebank corpus, which is a SyntaxCorpusReader Object.
    Assumed to have words(), tagged_words(), sents(), tagged_sents() methods. 
    May require adjustments for other corpora.

    Tools:
    """

    def __init__(self, corpus_reader):
        """
        Initialize object.

        SyntaxCorpusReader corpus_reader: treebank corpus or other SyntaxCorpusReader object
        Tested on NLTK's Sinica Treebank corpus, which is a SyntaxCorpusReader Object.
        Assumed to have words(), tagged_words(), sents(), tagged_sents() methods. 
        May require adjustments for other corpora.
        """
        
        #POS-tagged words
        self.tagged_words = corpus_reader.tagged_words()

        #untagged words
        self.words = corpus_reader.words()

        #list of unique Chinese characters (hanzi)
        self.hanzi = self.get_hanzi()

        #dictionary of format {char:[compounds]}, e.g. {'好':['你好', '好奇', ...]}
        self.compound_dict = self.get_compound_dict()

        #list of tuples of format [(char, # of compounds with that character)]
        self.productivity_list = self.get_productivity_list()

    def get_hanzi(self):
        """
        Create list of unique Chinese characters that appear in the corpus

        Returns a list of Chinese characters. 
        """

        #list of unique words in corpus
        wordtypes = sorted(set(self.words))
        hanzi = []

        #iterate through the words and add to list of characters
        for wordtype in wordtypes:
            for char in wordtype:
                if char not in hanzi:
                    hanzi.append(char)
        return hanzi

    def get_wordtag_dict(self):
        """
        Create dictionary of format {'word':[tag1, tag2]}
        """
        wordtag_dict = defaultdict(list)

        #iterate through tagged words
        for word, tag in set(self.tagged_words):
            #a word might have multiple tags
            if tag not in wordtag_dict[word]:
                wordtag_dict[word].append(tag)
        return wordtag_dict

    def get_tagword_dict(self):
        """
        Create dictionary of format {'tag':[word1, word2]}
        """

        tagword_dict = defaultdict(list)

        #iterate through tagged words
        for word, tag in set(self.tagged_words):
            #each tag will certainly have more than one word
            if word not in tagword_dict[tag]:
                tagword_dict[tag].append(word)
        return tagword_dict

    def get_words_by_tag(self, tag_query):
        """
        Return a list of words with a particular tag

        String tag_query: the tag to look for, depending on the corpus's tag conventions
        """
        words_by_tag = [word for word, tag in set(self.tagged_words) if tag == tag_query]
        return words_by_tag

    def get_compound_dict(self):
        """
        Create dictionary of format {char:[compounds]}, e.g. {'好':['你好', '好奇', ...]}
        """
        compound_dict = defaultdict(list)

        #iterate through untagged words
        for word in set(self.words):
            for char in word:
                if word not in compound_dict[char]:
                    compound_dict[char].append(word)
        return compound_dict

    def get_productivity_list(self):
        """
        Create list of tuples with format [(char, # compounds with that character)]
        """
        productivity_list = []
        #iterate over dictionary created by get_compound_dict()
        for key in self.compound_dict.keys():
            productivity_list.append((key,len(self.compound_dict[key])))

        #return list sorted from most to least number of compounds
        return sorted(productivity_list, key=lambda x:x[1], reverse=True)

class SinicaVec:
    """
    Class for analyzing compound words in Chinese treebank data using gensim's Word2Vec implementation.

    Required for init: sentence-tokenized Chinese corpus data.

    Tools:
    """

    def __init__(self, corpus_reader):
        """
        Required for init: sentence-tokenized Chinese corpus data
        """
        self.sents = corpus_reader.sents()

        #min_count is low because of very small sample corpus size
        self.model = Word2Vec(self.sents, min_count=1)

        #dictionary of format {char:[compounds]}, e.g. {'好':['你好', '好奇', ...]}
        self.compound_dict = self.get_compound_dict()

    def get_compound_dict(self):
        """
        Create dictionary of format {char:[compounds]}, e.g. {'好':['你好', '好奇', ...]}
        """
        compound_dict = defaultdict(list)
        for word in set(self.model.vocab):
            for char in word:
                if word not in compound_dict[char]:
                    compound_dict[char].append(word)
        return compound_dict

    def get_compound_similarity_list(self, query):
        """
        Returns similarities of query with all words sharing roots (語素)  with query

        E.g. using method on "建造" returns a list of similarity scores
        comparing "建造" with "造成", "建立", "人造雨" etc.

        String query: word to split up into roots (e.g.) "建造" splits into "建" and "造",
        then finds all words in model that have either "建" or "造" or both in them.

        Return type: [(word1, similarity_to_query), (word2, similarity_to_query)]
        """
        shared_roots = []
        roots = [char for char in query if char in self.compound_dict.keys()]
        for root in roots:
            for compound in self.compound_dict[root]:
                shared_roots.append(compound)
        #create a list of all unique words that share roots
        shared_types = list(set(shared_roots))

        similarity_list = []
        for compound in shared_types:
            similarity_list.append((compound, self.model.similarity(query, compound)))
        return sorted(similarity_list, key=lambda x: x[1], reverse=True)

    def get_avg_compound_similarity(self, query):
        """
        Returns on average how similar the word is to all other words which share its roots.

        String query: word to split up into roots (e.g.) "建造" splits into "建" and "造",
        then finds all words in model that have either "建" or "造" or both in them.
        Then it gets similarity scores between 建造 and each of these words, 
        and returns the average.

        Return type: int
        """
        sl = self.get_compound_similarity_list(query)
        return (sum([item[1] for item in sl]) / len(sl))

    def get_avg_similarity_dict(self, min_entries = 1):
        """
        Returns a dictionary of the form {word:avg similarity of word with other words containing its roots}

        int min_entries: minimum number of entries in the compound similarity list.
        Increase this value to exclude low-frequency words that will
        likely have a very high average similarity (because we might only be comparing them
        to themselves)

        Return type: dict of format {word:avg_compound_similarity}
        """
        avg_sim_dict = {word:self.get_avg_compound_similarity(word) for word in self.model.vocab \
            if len(self.get_compound_similarity_list(word)) >= min_entries}
        return avg_sim_dict

    def get_most_similar(self, min_entries = 1):
        avg_sim_list = [(word, self.get_avg_compound_similarity(word)) for word in self.model.vocab \
            if len(self.get_compound_similarity_list(word)) >= min_entries]
        return sorted(avg_sim_list, key=lambda x: x[1], reverse=True)

    def get_max_compound_similarity(self):
        """
        Returns the most internally consistent word, 
         i.e. the word that is the most similar to all words sharing its roots
        """
        avg_sim_dict = self.get_avg_similarity_dict(min_entries = 2)
        max_avg_sim = max(avg_sim_dict.values())
        max_similarity = [(key, avg_sim_dict[key]) for key in avg_sim_dict.keys() if avg_sim_dict[key] == max_avg_sim]
        return max_similarity

    def get_min_compound_similarity(self):
        """
        Returns the least internally consistent word, 
        i.e. the word that is the least similar to all words sharing its roots.
        """
        avg_sim_dict = self.get_avg_similarity_dict(min_entries = 2)
        min_avg_sim = min(avg_sim_dict.values())
        min_similarity = [(key, avg_sim_dict[key]) for key in avg_sim_dict.keys() if avg_sim_dict[key] == min_avg_sim]
        return min_similarity

class CDict:
    """
    Class for analyzing compound words using open-source CC-CEDICT dictionary.

    Required for init: CC-CEDICT dictionary in CSV format.
    
    NOTE: If you have a copy of the dictionary in the standard .u8 format distributed
    from the website, you can use the from_txt method to convert it.
    """
    def __init__(self, cedict_source):

        #dictionary of the format {character:[list of all dictionary entries for words including that character]}
        self.word_dict = defaultdict(list)

        #list of all unique Chinese characters in the dictionary
        self.hanzi = []

        #open the CSV dictionary
        with open(cedict_source, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                #add unique Chinese characters to self.hanzi
                if len(row[0]) == 1 and row[0] not in self.hanzi: self.hanzi.append(row[0])
                #populate self.word_dict in list form because a word might have more than one entry
                self.word_dict[row[0]].append(row[0:])

        #Length of longest entry; used for some methods
        self.max_entry_len = max([len(key) for key in self.word_dict.keys()])

    @classmethod
    def from_txt(self, txt, dest):
        """
        Convert CC-CEDICT to CSV format.

        String txt: path of CC-CEDICT source file
        String dest: path of destination file

        Assumes entries organized as follows:
        trad simp [pinyin] /meaning1/meaning2/meaning3/meaning4/

        Example:
        語言學 语言学 [yu3 yan2 xue2] /linguistics/
        """
        csv_header = 'trad,simp,pinyin,def1,def2,def3,def4,def5,'+\
            'def6,def7,def8,def9,def10,def11,def12,'+\
            'def13,def14,def15,def16,def17,def18,def19,def20,def21,def22'
        #The way the definitions are delineated
        pattern1 = r'(\ \[)|(\] \/)|\/'

        #Open source file
        cedict_source = open(txt, 'r').read()

        #Open or create destination file
        cedict_dest = open(dest, 'w')

        #Remove comments from the beginning of the file
        result0 = re.sub(r'#[^\n]*\n', r'', cedict_source)

        #Replace definition delineation with commas
        result1 = re.sub(pattern1, ',', result0)

        #Split lines
        result2 = re.sub('/\n', '\n', result1)

        #The way the traditional characters, simplified characters, and pinyin are delineated
        #NOTE: This regex is from a summer project, and I didn't come up with it completely on my own. 
        #I figured out the individual elements but got help putting them together
        result3 = re.sub(r"([\u3300-\u9fff]+)\s+(?:[a-z]+)?([\u3300-\u9fff]+)", r"\1,\2", result2, 0, re.IGNORECASE)

        #Write the CSV header
        cedict_dest.write(csv_header)

        #Write to the destination file
        cedict_dest.write(result3)

        #Close the file
        cedict_dest.close()    

    def search(self, query):
        """
        Search for words matching the query.

        String query: the word(s) to search for.
 
        Return type: list of CC-CEDICT entries

        '*' is a wildcard.
        '而*' matches words starting with '而': '而且', etc. 
        '*而' matches words ending with '而': '因而', etc.
        '*而*'matches words that have '而' in the middle: '不翼而飛', etc.
        """

        #if first and last char in query is *, find all words with query in the middle
        if query.startswith('*') and query.endswith('*'):
            result = [self.word_dict[key] for key in self.word_dict.keys() if query[1:-1] in key \
                and not key.startswith(query[1:-1]) and not key.endswith(query[1:-1])]

        #if first char in query is *, find all words ending with the query
        elif query.startswith('*'):
            result = [self.word_dict[key] for key in self.word_dict.keys() if key.endswith(query[1:])]

        #if last char in query is *, find all words starting with the query
        elif query.endswith('*'):
            result = [self.word_dict[key] for key in self.word_dict.keys() if key.startswith(query[:-1])]

        #if no * in query, find all words starting with the query
        else:
            result = [self.word_dict[key] for key in self.word_dict.keys() if key.startswith(query)]
        return result

    def get_compounds(self, query):
        """
        Returns a list of compounds including the query
        """
        return [result[0][0] for result in self.search(query)]        

    def get_idioms(self):
        """
        Returns list of idioms (chengyu) in the dictionary
        """
        idioms = []
        for key in self.word_dict.keys():
            is_idiom = False
            for entry in self.word_dict[key]:
                for item in entry:
                    if '(idiom)' in item:
                        is_idiom = True
            if is_idiom: idioms.append(key)
        return idioms

    def get_variants(self):
        """
        Returns list of character variants (archaic, etc.) in the dictionary
        """
        variants = []
        for zi in self.hanzi:
            is_variant = False
            for entry in self.word_dict[zi]:
                for item in entry:
                    if 'variant' in item:
                        is_variant = True
            if is_variant: variants.append(zi)
        return variants

    def get_gugja(self):
        """
        Returns list of gugja (special hanzi for Korean) in the dictionary
        """
        gugja = []
        for zi in self.hanzi:
            is_gugja = False
            for entry in self.word_dict[zi]:
                for item in entry:
                    if 'gugja' in item:
                        is_gugja = True
            if is_gugja: gugja.append(zi)
        return gugja

    def get_nonrare(self):
        """
        Returns list of non-rare characters in the dictionary
        """
        hanzi_set = set(self.hanzi)
        variants_set = set(self.get_variants())
        gugja_set = set(self.get_gugja())
        rare_set = variants_set | gugja_set
        nonrare_set = hanzi_set - rare_set
        return nonrare_set
