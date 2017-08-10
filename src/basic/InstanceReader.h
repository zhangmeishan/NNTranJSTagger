#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "N3LDG.h"
#include "Utf.h"
#include <sstream>

using namespace std;

class InstanceReader : public Reader {
  public:
    InstanceReader() {
    }
    ~InstanceReader() {
    }

    Instance *getNext() {
        m_instance.clear();
        string strLine;
        while (1) {
            if (!my_getline(m_inf, strLine)) {
                break;
            }
            if (!strLine.empty())
                break;
        }

        vector<string> wordInfo;
        split_bychar(strLine, wordInfo, ' ');

        string sentence = "";
        vector<string> words, tags;
        vector<string> splits;
        words.resize(wordInfo.size());
        tags.resize(wordInfo.size());
        for (int i = 0; i < wordInfo.size(); ++i) {
            split_bychar(wordInfo[i], splits, '_');
            if (splits.size() != 2) {
                std::cout << "error: " << strLine << std::endl;
                return getNext();
            }
            words[i] = splits[0];
            tags[i] = splits[1];
            sentence = sentence + splits[0];
        }

        vector<string> charInfo;
        getCharactersFromUTF8String(sentence, charInfo);

        m_instance.allocate(wordInfo.size(), charInfo.size());
        for (int i = 0; i < wordInfo.size(); ++i) {
            m_instance.words[i] = words[i];
            m_instance.tags[i] = tags[i];
        }
        for (int i = 0; i < charInfo.size(); ++i) {
            m_instance.chars[i] = charInfo[i];
        }

        return &m_instance;
    }
};

#endif

