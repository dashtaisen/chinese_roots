#Nicholas A Miller
#nicholas.anthony.miller@gmail.com
#19 Dec 2016
#Chinese words and roots toolkit

Description:
=====================
Toolbox for using CC-CEDICT and NLTK's Sinica Treebank sample to work with Mandarin words (ci2) and roots (yu3su4).

Files included
===================================
- project.py: the actual code
- project_test.py: a test suite
- project_test_result.txt: sample result from running test suite
- cedict.csv: CC-CEDICT, already converted to CSV, for CDict class
- README.md: this README file
- nmiller_ling131_project_presentation: presentation slides for Monday 19 Dec.


Setup instructions:
====================

Required modules
----------------
For the module to run, the following need to be installed for the version of Python you're using:
- gensim
- nltk

For the CDict class to work, you can either have the (provided) CC-CEDICT.csv file saved to the same directory as the project file,
or you can download a copy of CC-CEDICT from the CC-CEDICT website and run the
CDict.from_txt() method from project.py.

Running instructions:
=====================

Option 1: 
-----------------

Run project_test.py to see what the toolbox can do. 
(Unix terminal example) 
    python project_test.py > project_test_result.txt

**NOTE: The whole test suite is divided into 3 parts, which are labeled in project_test.py. If the code is too slow, you can comment out one or more parts of the test suite.**

Option 2: 
------------------
Open your interpreter (e.g. IDLE), create objects of each class, and play with their methods.
    import nltk
    from nltk.corpus import sinica_treebank as sinica
    import project

    project_pos = project.SinicaPOS(sinica)
    project_vec = project.SinicaVec(sinica)
    cedict_source = './cedict.csv'
    project_cdict = project.CDict(cedict_source)
