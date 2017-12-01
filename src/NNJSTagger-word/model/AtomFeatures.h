/*
 * AtomFeatures.h
 *
 *  Created on: Aug 25, 2016
 *      Author: mszhang
 */

#ifndef SRC_AtomFeatures_H_
#define SRC_AtomFeatures_H_
#include "ModelParams.h"
struct AtomFeatures {
  public:
    string str_word;
    string str_tag;
    string str_len;
    int next_position;
    int char_size;
    int word_num;
    int word_start;
  public:
    IncLSTM1Builder* p_word_lstm;
    LSTM1Builder* p_char_left_lstm;
    LSTM1Builder* p_char_right_lstm;
  public:
    void clear() {
        str_word = "";
        str_tag = "";
        str_len = "";
        next_position = -1;
        char_size = -1;
        word_num = -1;
        word_start = -1;
        p_word_lstm = NULL;
        p_char_left_lstm = NULL;
        p_char_right_lstm = NULL;
    }

};

#endif /* SRC_AtomFeatures_H_ */
