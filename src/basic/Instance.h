#ifndef _JST_INSTANCE_
#define _JST_INSTANCE_

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include "Metric.h"
#include "Utf.h"

using namespace std;

class Instance {
  public:
    Instance() {
    }
    ~Instance() {
    }

    int wordsize() const {
        return words.size();
    }

    int charsize() const {
        return chars.size();
    }

    void clear() {
        words.clear();
        chars.clear();
        tags.clear();
    }

    void allocate(int length, int charLength) {
        clear();
        words.resize(length);
        chars.resize(charLength);
        tags.resize(length);
    }

    void copyValuesFrom(const Instance& anInstance) {
        allocate(anInstance.wordsize(), anInstance.charsize());
        for (int i = 0; i < anInstance.wordsize(); i++) {
            words[i] = anInstance.words[i];
            tags[i] = anInstance.tags[i];
        }
        for (int i = 0; i < anInstance.charsize(); i++) {
            chars[i] = anInstance.chars[i];
        }
    }


    void evaluate(const vector<string>& resulted_segs, Metric& eval) const {
        unordered_set<string> golds;
        getSegIndexes(words, golds);

        unordered_set<string> preds;
        getSegIndexes(resulted_segs, preds);

        unordered_set<string>::iterator iter;
        eval.overall_label_count += golds.size();
        eval.predicated_label_count += preds.size();
        for (iter = preds.begin(); iter != preds.end(); iter++) {
            if (golds.find(*iter) != golds.end()) {
                eval.correct_label_count++;
            }
        }
    }

    void evaluate(const vector<string>& resulted_segs, const vector<string>& resulted_tags, Metric& eval, Metric& tageval) const {
        unordered_set<string> golds;
        getSegIndexes(words, golds);

        unordered_set<string> preds;
        getSegIndexes(resulted_segs, preds);

        unordered_set<string>::iterator iter;
        eval.overall_label_count += golds.size();
        eval.predicated_label_count += preds.size();
        for (iter = preds.begin(); iter != preds.end(); iter++) {
            if (golds.find(*iter) != golds.end()) {
                eval.correct_label_count++;
            }
        }


        unordered_set<string> tag_golds;
        getSegIndexes(words, tags, tag_golds);

        unordered_set<string> tag_preds;
        getSegIndexes(resulted_segs, resulted_tags, tag_preds);

        tageval.overall_label_count += tag_golds.size();
        tageval.predicated_label_count += tag_preds.size();
        for (iter = tag_preds.begin(); iter != tag_preds.end(); iter++) {
            if (tag_golds.find(*iter) != tag_golds.end()) {
                tageval.correct_label_count++;
            }
        }
    }

    void getSegIndexes(const vector<string>& segs, unordered_set<string>& segIndexes) const {
        segIndexes.clear();
        int idx = 0, idy = 0;
        string curWord = "";
        int beginId = 0;
        int target_length = getUTF8StringLength(segs[idy]);
        int source_length = 0;
        while (idx < chars.size() && idy < segs.size()) {
            curWord = curWord + chars[idx];
            source_length++;
            if (source_length == target_length) {
                stringstream ss;
                ss << "[" << beginId << "," << idx << "]";
                segIndexes.insert(ss.str());
                beginId = idx + 1;
                curWord = "";
                source_length = 0;
                idx++;
                idy++;
                if (idy < segs.size())target_length = getUTF8StringLength(segs[idy]);
                else break;
            } else {
                idx++;
            }
        }

        if (idx != chars.size() || idy != segs.size()) {
            std::cout << "error segs, please check" << std::endl;
        }
    }

    void getSegIndexes(const vector<string>& segs, const vector<string>& labels, unordered_set<string>& segIndexes) const {
        segIndexes.clear();
        int idx = 0, idy = 0;
        string curWord = "";
        int beginId = 0;
        int target_length = getUTF8StringLength(segs[idy]);
        int source_length = 0;
        while (idx < chars.size() && idy < segs.size()) {
            curWord = curWord + chars[idx];
            source_length++;
            if (source_length == target_length) {
                stringstream ss;
                ss << "[" << beginId << "," << idx << "]" + labels[idy];
                segIndexes.insert(ss.str());
                beginId = idx + 1;
                curWord = "";
                source_length = 0;
                idx++;
                idy++;
                if (idy < segs.size())target_length = getUTF8StringLength(segs[idy]);
                else break;
            } else {
                idx++;
            }
        }

        if (idx != chars.size() || idy != segs.size()) {
            std::cout << "error segs, please check" << std::endl;
        }
    }



  public:
    vector<string> words;
    vector<string> chars;
    vector<string> tags;

};

#endif

