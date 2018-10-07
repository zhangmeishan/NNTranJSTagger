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
    vector<LookupNode> ext_char_inputs;
    vector<LookupNode> chartype_inputs;
    vector<LookupNode> bichar_inputs;
    vector<LookupNode> ext_bichar_inputs;
    LookupNode bichar_start_input, bichar_end_input;
    LookupNode ext_bichar_start_input, ext_bichar_end_input;
    vector<ConcatNode> char_left_represents;
    vector<ConcatNode> char_right_represents;
    vector<UniNode> char_left_conv;
    vector<UniNode> char_right_conv;
    LSTM1Builder char_left_lstm;
    LSTM1Builder char_right_lstm;

  public:
    inline void resize(int max_sentence_length) {
        char_inputs.resize(max_sentence_length);
        ext_char_inputs.resize(max_sentence_length);
        bichar_inputs.resize(max_sentence_length);
        ext_bichar_inputs.resize(max_sentence_length);
        chartype_inputs.resize(max_sentence_length);
        char_left_represents.resize(max_sentence_length);
        char_right_represents.resize(max_sentence_length);
        char_left_conv.resize(max_sentence_length);
        char_right_conv.resize(max_sentence_length);
        char_left_lstm.resize(max_sentence_length);
        char_right_lstm.resize(max_sentence_length);
    }

  public:
    inline void initial(ModelParams& params, HyperParams& hyparams) {
        int length = char_inputs.size();
        for (int idx = 0; idx < length; idx++) {
            char_inputs[idx].setParam(&params.char_table);
            bichar_inputs[idx].setParam(&params.bichar_table);
            ext_char_inputs[idx].setParam(&params.ext_char_table);
            ext_bichar_inputs[idx].setParam(&params.ext_bichar_table);
            chartype_inputs[idx].setParam(&params.chartype_table);
            char_left_conv[idx].setParam(&params.char_tanh_conv);
            char_right_conv[idx].setParam(&params.char_tanh_conv);
        }
        bichar_start_input.setParam(&params.bichar_table);
        bichar_end_input.setParam(&params.bichar_table);
        ext_bichar_start_input.setParam(&params.ext_bichar_table);
        ext_bichar_end_input.setParam(&params.ext_bichar_table);

        char_left_lstm.init(&params.char_left_lstm, hyparams.dropProb, true);
        char_right_lstm.init(&params.char_right_lstm, hyparams.dropProb, false);

        for (int idx = 0; idx < length; idx++) {
            char_inputs[idx].init(hyparams.char_dim, hyparams.dropProb);
            bichar_inputs[idx].init(hyparams.bichar_dim, hyparams.dropProb);
            ext_char_inputs[idx].init(hyparams.char_dim, hyparams.dropProb);
            ext_bichar_inputs[idx].init(hyparams.bichar_dim, hyparams.dropProb);
            chartype_inputs[idx].init(hyparams.chartype_dim, hyparams.dropProb);
            char_left_represents[idx].init(hyparams.char_concat_dim, -1);
            char_right_represents[idx].init(hyparams.char_concat_dim, -1);
            char_left_conv[idx].init(hyparams.char_hidden_dim, hyparams.dropProb);
            char_right_conv[idx].init(hyparams.char_hidden_dim, hyparams.dropProb);
        }

        bichar_start_input.init(hyparams.bichar_dim, hyparams.dropProb);
        bichar_end_input.init(hyparams.bichar_dim, hyparams.dropProb);

        ext_bichar_start_input.init(hyparams.bichar_dim, hyparams.dropProb);
        ext_bichar_end_input.init(hyparams.bichar_dim, hyparams.dropProb);
    }


  public:
    inline void forward(Graph* cg, const std::vector<std::string>* pCharacters) {
        int char_size = pCharacters->size();
        string unichar, biChar, chartype;
        for (int idx = 0; idx < char_size; idx++) {
            unichar = (*pCharacters)[idx];
            char_inputs[idx].forward(cg, unichar);
            ext_char_inputs[idx].forward(cg, unichar);

            chartype = wordtype(unichar);
            chartype_inputs[idx].forward(cg, chartype);

            if (idx < char_size - 1) {
                biChar = (*pCharacters)[idx] + (*pCharacters)[idx + 1];
                bichar_inputs[idx].forward(cg, biChar);
                ext_bichar_inputs[idx].forward(cg, biChar);
            }
        }

        bichar_start_input.forward(cg, nullkey + (*pCharacters)[0]);
        bichar_end_input.forward(cg, (*pCharacters)[char_size-1] + nullkey);
        ext_bichar_start_input.forward(cg, nullkey + (*pCharacters)[0]);
        ext_bichar_end_input.forward(cg, (*pCharacters)[char_size - 1] + nullkey);

        for (int idx = 0; idx < char_size; idx++) {
            if (idx == 0) {
                char_left_represents[idx].forward(cg, &(char_inputs[idx]), &bichar_start_input, &(ext_char_inputs[idx]), &ext_bichar_start_input, &(chartype_inputs[idx]));
            } else {
                char_left_represents[idx].forward(cg, &(char_inputs[idx]), &(bichar_inputs[idx - 1]), &(ext_char_inputs[idx]), &(ext_bichar_inputs[idx - 1]), &(chartype_inputs[idx]));
            }
            char_left_conv[idx].forward(cg, &(char_left_represents[idx]));
        }

        for (int idx = 0; idx < char_size; idx++) {
            if (idx == char_size - 1) {
                char_right_represents[idx].forward(cg, &(char_inputs[idx]), &bichar_end_input, &(ext_char_inputs[idx]), &ext_bichar_end_input, &(chartype_inputs[idx]));
            } else {
                char_right_represents[idx].forward(cg, &(char_inputs[idx]), &(bichar_inputs[idx]), &(ext_char_inputs[idx]), &(ext_bichar_inputs[idx]), &(chartype_inputs[idx]));
            }
            char_right_conv[idx].forward(cg, &(char_right_represents[idx]));
        }

        char_left_lstm.forward(cg, getPNodes(char_left_conv, char_size));
        char_right_lstm.forward(cg, getPNodes(char_right_conv, char_size));
    }

};

#endif /* SRC_GlobalNodes_H_ */
