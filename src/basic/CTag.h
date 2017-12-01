/*
 * CTag.h
 *
 *  Created on: Aug 12, 2017
 *      Author: mszhang
 */

#ifndef BASIC_CTAG_H_
#define BASIC_CTAG_H_
#include <unordered_map>
#include <unordered_set>

class CTag {
  public:
    const string PENN_TAG_STRINGS[33] = {
        "NN", "VV",
        "NR", "AD",
        "P", "CD", "M", "JJ",
        "DEC", "DEG",
        "NT", "CC", "VA", "LC",
        "PN", "DT", "VC", "AS", "VE",
        "OD", "IJ", "ON",
        "ETC", "MSP", "CS", "BA",
        "DEV", "SB", "SP", "LB",
        "FW", "DER", "PU",
    };

    const bool PENN_TAG_CLOSED[33] = {
        false, false,
        false, false,
        true, false, false, false,
        true, true,
        false, true, false, true,
        true, true, true, true, true,
        false, true, true,
        true, true, true, true,
        true, true, true, true,
        false, true, true,
    };

  public:
    CTag() {
        tag_to_id.clear();
        for (int idx = 0; idx < 33; idx++) {
            tag_to_id[PENN_TAG_STRINGS[idx]] = idx;
        }
        firstchar_pos.clear();
        word_pos.clear();
        char_freq.clear();
        word_freq.clear();

        maxFreqChar = -1;
        maxFreqWord = -1;
    }


  public:
    inline string getTagName(int tagId) {
        if (tagId < 33 && tagId >= 0) {
            return PENN_TAG_STRINGS[tagId];
        }
        return nullkey;
    }

    inline int size() {
        return 33;
    }

    inline int getTagId(const string& tagName) {
        if (tag_to_id.find(tagName) != tag_to_id.end()) {
            return tag_to_id[tagName];
        }
        return -1;
    }

    inline bool isClosedTag(int tagId) {
        if (tagId < 33 && tagId >= 0) {
            return PENN_TAG_CLOSED[tagId];
        }
        return false;
    }

    inline bool canAssignWordTag(string curWord, int tagId) {
        // curChar seen in training data
        if (word_pos.find(curWord) != word_pos.end()) {
            //curchar pos pair seen in training data
            if (word_pos[curWord].find(tagId) != word_pos[curWord].end()) {
                return true;
            }

            //curchar pos pair unseen
            //condition 1: tagId is a closed tag, can not
            if (PENN_TAG_CLOSED[tagId] == true) {
                return false;
            }

            //condition 2: curChar is high frequency, can not
            if (word_freq[curWord] > (maxFreqWord / 5000 + 3)) {
                return false;
            }
            //other
            return true;
        }

        // curChar not seen
        //condition 1: tagId is a closed tag, can not
        if (PENN_TAG_CLOSED[tagId] == true) {
            return false;
        }

        //other
        return true;

    }

    inline bool canAssignCharTag(string curChar, int tagId) {
        // curChar seen in training data
        if (firstchar_pos.find(curChar) != firstchar_pos.end()) {
            //curchar pos pair seen in training data
            if (firstchar_pos[curChar].find(tagId) != firstchar_pos[curChar].end()) {
                return true;
            }

            //curchar pos pair unseen
            //condition 1: tagId is a closed tag, can not
            if (PENN_TAG_CLOSED[tagId] == true) {
                return false;
            }

            //condition 2: curChar is high frequency, can not
            if (char_freq[curChar] > (maxFreqChar / 5000 + 5)) {
                return false;
            }
            //other
            return true;
        }

        //curChar not seen
        //condition 1: tagId is a closed tag, can not
        if (PENN_TAG_CLOSED[tagId] == true) {
            return false;
        }

        //other
        return true;
    }

    inline void clear() {
        tag_to_id.clear();
        firstchar_pos.clear();
        word_pos.clear();
        char_freq.clear();
        word_freq.clear();
        int maxFreqChar = -1;
        int maxFreqWord = -1;
    }


  public:
    unordered_map<string, int> tag_to_id;
    unordered_map<string, unordered_set<int> > firstchar_pos;
    unordered_map<string, unordered_set<int> > word_pos;
    unordered_map<string, int> char_freq;
    unordered_map<string, int> word_freq;
    int maxFreqChar;
    int maxFreqWord;
};

#endif /* BASIC_TAG_H_ */
