/*
 * Feature.h
 *
 *  Created on: Aug 25, 2016
 *      Author: mszhang
 */

#ifndef SRC_GlobalNodes_H_
#define SRC_GlobalNodes_H_

#include "ModelParams.h"

struct GlobalNodes {
    vector<LookupNode> char_inputs;
    vector<LookupNode> chartype_inputs;
    vector<ConcatNode> char_represents;
    LSTM1Builder char_left_lstm;
    LSTM1Builder char_right_lstm;

    vector<LookupNode> bichar_inputs;
    LSTM1Builder bichar_left_lstm;
    LSTM1Builder bichar_right_lstm;

  public:
    inline void resize(int max_sentence_length) {
        char_inputs.resize(max_sentence_length);
        chartype_inputs.resize(max_sentence_length);
        char_represents.resize(max_sentence_length);
        char_left_lstm.resize(max_sentence_length);
        char_right_lstm.resize(max_sentence_length);

        bichar_inputs.resize(max_sentence_length);
        bichar_left_lstm.resize(max_sentence_length);
        bichar_right_lstm.resize(max_sentence_length);
    }

  public:
    inline void initial(ModelParams& params, HyperParams& hyparams) {
        int length = char_inputs.size();
        for (int idx = 0; idx < length; idx++) {
            char_inputs[idx].setParam(&params.char_table);
            chartype_inputs[idx].setParam(&params.chartype_table);
            bichar_inputs[idx].setParam(&params.bichar_table);
        }

        char_left_lstm.init(&params.char_left_lstm, hyparams.dropProb, true);
        char_right_lstm.init(&params.char_right_lstm, hyparams.dropProb, false);

        bichar_left_lstm.init(&params.bichar_left_lstm, hyparams.dropProb, true);
        bichar_right_lstm.init(&params.bichar_right_lstm, hyparams.dropProb, false);

        for (int idx = 0; idx < length; idx++) {
            char_inputs[idx].init(hyparams.char_dim, hyparams.dropProb);
            bichar_inputs[idx].init(hyparams.bichar_dim, hyparams.dropProb);
            chartype_inputs[idx].init(hyparams.chartype_dim, hyparams.dropProb);
            char_represents[idx].init(hyparams.char_concat_dim, -1);
        }
    }


  public:
    inline void forward(Graph* cg, const std::vector<std::string>* pCharacters) {
        int char_size = pCharacters->size();
        string unichar, biChar, chartype;
        for (int idx = 0; idx < char_size; idx++) {
            unichar = (*pCharacters)[idx];
            char_inputs[idx].forward(cg, unichar);

            chartype = wordtype(unichar);
            chartype_inputs[idx].forward(cg, chartype);

            if (idx < char_size - 1) {
                biChar = (*pCharacters)[idx] + (*pCharacters)[idx + 1];
                bichar_inputs[idx].forward(cg, biChar);
            }
        }

        for (int idx = 0; idx < char_size; idx++) {
            char_represents[idx].forward(cg, &(char_inputs[idx]), &(chartype_inputs[idx]));
        }

        char_left_lstm.forward(cg, getPNodes(char_represents, char_size));
        char_right_lstm.forward(cg, getPNodes(char_represents, char_size));

        bichar_left_lstm.forward(cg, getPNodes(bichar_inputs, char_size - 1));
        bichar_right_lstm.forward(cg, getPNodes(bichar_inputs, char_size - 1));
    }

};

#endif /* SRC_GlobalNodes_H_ */
