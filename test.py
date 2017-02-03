"""
Name: Nicholas A Miller
Email: nicholas.anthony.miller@gmail.com
Description: Chinese words and roots toolkit
"""

import nltk
from nltk.corpus import sinica_treebank as sinica
from collections import defaultdict

import chinese_roots

#PART 0: preliminaries
#replace this line with wherever your CSV version of the CC-CEDICT is
cedict_source = '../project/cedict.csv'


#PART 1: working with SinicaPOS
print("Demonstrating SinicaPOS class")
pos_dict = chinese_roots.SinicaPOS(sinica)

#some introductory statistics and comparisons
print("Number of word tokens in Sinica Treebank sample: {0}".format(len(pos_dict.words)))
print("Number of word types in Sinica Treebank sample: {0}".format(len(set(pos_dict.words))))
print("Number of sentences in Sinica Treebank sample: {0}".format(len(sinica.sents())))
print("Number of distinct Chinese characters in Sinica Treebank sample: {0}".format(len(pos_dict.hanzi)))

#Create dictionary of format {'word':[tag1, tag2]}
word_tag_dict = pos_dict.get_wordtag_dict()

#show a sample of word-tag dict
print("Sample of word-tag dict:")
sample_keys = list(word_tag_dict.keys())[:2]
for sample_key in sample_keys:
    print(word_tag_dict[sample_key])

#Create dictionary of format {'tag':[word1, word2]}
tag_word_dict = pos_dict.get_tagword_dict()

#show a sample of tag-word dict
print("Sample words for each POS:")
pos_keys = sorted(list(tag_word_dict.keys()))
for key in pos_keys:
    print("{0}:{1}".format(key, tag_word_dict[key][:2]))

#getting words by tag
print("Demonstrate getting words by tag (Dj--interrogatives)")
interrogatives = pos_dict.get_words_by_tag('Dj')
print(interrogatives[:5])

#get list of proper names in corpus
print("Use get_words_by_tag() to find proper names in corpus")
names = set(pos_dict.get_words_by_tag('Nba')) | set(pos_dict.get_words_by_tag('Nbc')) 
print("Corpus sample contains {0} proper names".format(len(names)))

#Which word(s) generate the greatest number of compounds?
print("Words with greatest number of compounds:")
print(pos_dict.get_productivity_list()[:10])

#Which word(s) generate the greatest number of compounds that aren't proper names?
word_set = set(pos_dict.words)
name_set = names
non_names = list(word_set - name_set)
non_name_compound_dict = defaultdict(list)
for word in non_names:
    for char in word:
        if word not in non_name_compound_dict[char]:
            non_name_compound_dict[char].append(word)
non_name_productivity_list = [(word, len(non_name_compound_dict[word])) for word in non_name_compound_dict.keys()]
print("Words with greatest number of compounds that aren't names:")
print(sorted(non_name_productivity_list, key=lambda x: x[1], reverse=True)[:10])

#Which words have the most possible parts of speech?
max_tags = max([len(value) for value in word_tag_dict.values()])
most_tags = [key for key in word_tag_dict.keys() if len(word_tag_dict[key]) == max_tags]
print("Most possible parts of speech for a given word: {0}".format(max_tags))
print("Words with {0} possible parts of speech:".format(max_tags))
print(most_tags)
#END OF PART 1


#PART 2: working with SinicaVec
pos_vect = chinese_roots.SinicaVec(sinica)
print("Number of words detected by Word2Vec vocabulary model: {0}".format(len(pos_vect.model.vocab)))
query = '建造'
print("Demonstrating with {0}: ".format(query))
print("Searching for words that contain one or more of {0}:".format([char for char in query]))
csl = pos_vect.get_compound_similarity_list(query)
print(csl)

print("Average similarity between {0} and other words containing {1}: ".format(query, [char for char in query]))
print(pos_vect.get_avg_compound_similarity(query))

print("Words that are most similar to other words with which they share roots: ")
print(pos_vect.get_most_similar(5)[:10])

print("Words that are least similar to other words with which they share roots: ")
print(pos_vect.get_most_similar(5)[-10:])
#END OF PART 2


#PART 3: working with CDict

print("Creating CC-CEDICT compound dictionary")
cdict = chinese_roots.CDict(cedict_source)
print("Number of entries: {0}".format(len(cdict.word_dict.keys())))
print("Number of unique Chinese characters: {0}".format(len(cdict.hanzi)))
nonrare = cdict.get_nonrare()
print("Number of unique Chinese characters excluding variants and other rare characters: {0}".format(len(nonrare)))
print("Number of idioms (cheng2yu3): {0}".format(len(cdict.get_idioms())))
print("Length of longest key: {0}, {1}".format(cdict.max_entry_len, [key for key in cdict.word_dict.keys() \
       if len(key) == cdict.max_entry_len]))

#demonstrate searching
query1 = '建'
print("Sample of entries beginning with {0}:".format(query1))
query2 = '建*'
print(cdict.search(query2)[:5])
print("Sample of entries ending with {0}:".format(query1))
query3 = '*建'
print(cdict.search(query3)[:5])
print("Sample of entries with {0} in the middle:".format(query1))
query4 = '*建*'
print(cdict.search(query4)[:5])

#demonstrate compound list
query1 = '建'
print("Sample of compounds beginning with {0}:".format(query1))
query2 = '建*'
print(cdict.get_compounds(query2)[:5])
print("Sample of compounds ending with {0}:".format(query1))
query3 = '*建'
print(cdict.get_compounds(query3)[:5])
print("Sample of compounds with {0} in the middle:".format(query1))
query4 = '*建*'
print(cdict.get_compounds(query4)[:5])
#END OF PART 3

