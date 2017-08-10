#ifndef _CONLL_WRITER_
#define _CONLL_WRITER_

#include "Writer.h"
#include <sstream>

using namespace std;

class InstanceWriter : public Writer {
  public:
    InstanceWriter() {}
    ~InstanceWriter() {}
    int write(const Instance *pInstance) {
        if (!m_outf.is_open()) return -1;
        int word_count = pInstance->wordsize();
        if (word_count == 0) return -1;
        for (int i = 0; i < word_count - 1; ++i) {
            m_outf << pInstance->words[i] << "_" << pInstance->tags[i] << " ";
        }
        m_outf << pInstance->words[word_count - 1] << "_" << pInstance->tags[word_count - 1] << std::endl;
        return 0;
    }

    int write(const vector<string> &curWords, const vector<string> &curTags) {
        if (!m_outf.is_open()) return -1;
        int word_count = curWords.size();
        if (word_count == 0) return -1;
        for (int i = 0; i < word_count - 1; ++i) {
            m_outf << curWords[i] << "_" << curTags[i] << " ";
        }
        m_outf << curWords[word_count - 1] << "_" << curTags[word_count - 1] << std::endl;
        return 0;
    }

};

#endif

