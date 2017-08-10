/*
* Feature.h
*
*  Created on: Aug 25, 2016
*      Author: mszhang
*/

#ifndef SRC_ActionedNodes_H_
#define SRC_ActionedNodes_H_

#include "ModelParams.h"
#include "AtomFeatures.h"
#include "Action.h"

struct ActionedNodes {
    LookupNode last_word_input;
    LookupNode last_tag_input;
    ConcatNode word_concat;
    UniNode word_represent;
    IncLSTM1Builder word_lstm;
    ConcatNode word_state_concat;
    UniNode word_state_hidden;

    PSubNode char_span_repsent_left;
    PSubNode char_span_repsent_right;
    PSubNode bichar_span_repsent_left;
    PSubNode bichar_span_repsent_right;


    ConcatNode char_state_concat;
    UniNode char_state_hidden;

    UniNode app_state_represent;
    BiNode sep_state_represent;


    vector<LookupNode> current_action_input;
    vector<PDotNode> action_score;
    vector<PAddNode> outputs;

    BucketNode bucket_char, bucket_bichar, bucket_word;

    HyperParams *opt;

  public:
    ~ActionedNodes() {
    }
  public:
    inline void initial(ModelParams& params, HyperParams& hyparams) {
        last_word_input.setParam(&(params.ngram_char_table));
        last_word_input.init(hyparams.word_dim, hyparams.dropProb);
        last_tag_input.setParam(&(params.tag_table));
        last_tag_input.init(hyparams.tag_dim, hyparams.dropProb);
        word_concat.init(hyparams.word_concat_dim, -1);
        word_represent.setParam(&(params.word_represent));
        word_represent.init(hyparams.word_represent_dim, -1);
        word_lstm.init(&(params.word_lstm), hyparams.dropProb); //already allocated here
        word_state_concat.init(hyparams.word_feat_dim, -1);
        word_state_hidden.setParam(&(params.word_state_hidden));
        word_state_hidden.init(hyparams.word_state_dim, hyparams.dropProb);

        char_span_repsent_left.init(hyparams.char_lstm_dim, -1);
        char_span_repsent_right.init(hyparams.char_lstm_dim, -1);
        bichar_span_repsent_left.init(hyparams.bichar_lstm_dim, -1);
        bichar_span_repsent_right.init(hyparams.bichar_lstm_dim, -1);


        char_state_concat.init(hyparams.char_feat_dim, -1);
        char_state_hidden.setParam(&params.char_state_hidden);
        char_state_hidden.init(hyparams.char_state_dim, hyparams.dropProb);

        app_state_represent.setParam(&params.app_state_represent);
        app_state_represent.init(hyparams.state_hidden_dim, hyparams.dropProb);
        sep_state_represent.setParam(&params.sep_state_represent);
        sep_state_represent.init(hyparams.state_hidden_dim, hyparams.dropProb);

        current_action_input.resize(hyparams.action_num);
        action_score.resize(hyparams.action_num);
        outputs.resize(hyparams.action_num);

        //neural features
        for (int idx = 0; idx < hyparams.action_num; idx++) {
            current_action_input[idx].setParam(&(params.scored_action_table));
            current_action_input[idx].init(hyparams.state_hidden_dim, -1);

            action_score[idx].init(1, -1);
            outputs[idx].init(1, -1);
        }

        opt = &hyparams;

        bucket_char.init(hyparams.char_lstm_dim, -1);
        bucket_word.init(hyparams.word_lstm_dim, -1);
        bucket_bichar.init(hyparams.bichar_lstm_dim, -1);
    }


  public:
    inline void forward(Graph* cg, const vector<CAction>& actions, AtomFeatures& atomFeat, PNode prevStateNode) {
        vector<PNode> sumNodes;
        CAction ac;
        int ac_num;
        ac_num = actions.size();

        bucket_char.forward(cg, 0);
        bucket_bichar.forward(cg, 0);
        bucket_word.forward(cg, 0);
        PNode pseudo_char = &(bucket_char);
        PNode pseudo_bichar = &(bucket_bichar);
        PNode pseudo_word = &(bucket_word);


        //chars
        int char_posi = atomFeat.next_position;
        PNode char_node_left_curr = (char_posi  < atomFeat.char_size) ? &(atomFeat.p_char_left_lstm->_hiddens[char_posi]) : pseudo_char;
        PNode char_node_left_next1 = (char_posi + 1  < atomFeat.char_size) ? &(atomFeat.p_char_left_lstm->_hiddens[char_posi + 1]) : pseudo_char;
        PNode char_node_left_next2 = (char_posi + 2  < atomFeat.char_size) ? &(atomFeat.p_char_left_lstm->_hiddens[char_posi + 2]) : pseudo_char;
        PNode char_node_left_prev1 = (char_posi > 0) ? &(atomFeat.p_char_left_lstm->_hiddens[char_posi - 1]) : pseudo_char;
        PNode char_node_left_prev2 = (char_posi > 1) ? &(atomFeat.p_char_left_lstm->_hiddens[char_posi - 2]) : pseudo_char;

        PNode char_node_right_curr = (char_posi  < atomFeat.char_size) ? &(atomFeat.p_char_right_lstm->_hiddens[char_posi]) : pseudo_char;
        PNode char_node_right_next1 = (char_posi + 1  < atomFeat.char_size) ? &(atomFeat.p_char_right_lstm->_hiddens[char_posi + 1]) : pseudo_char;
        PNode char_node_right_next2 = (char_posi + 2  < atomFeat.char_size) ? &(atomFeat.p_char_right_lstm->_hiddens[char_posi + 2]) : pseudo_char;
        PNode char_node_right_prev1 = (char_posi > 0) ? &(atomFeat.p_char_right_lstm->_hiddens[char_posi - 1]) : pseudo_char;
        PNode char_node_right_prev2 = (char_posi > 1) ? &(atomFeat.p_char_right_lstm->_hiddens[char_posi - 2]) : pseudo_char;


        PNode char_lstm_left_start = atomFeat.word_start > 0 ? &(atomFeat.p_char_left_lstm->_hiddens[atomFeat.word_start - 1]) : pseudo_char;
        PNode char_lstm_left_end = char_node_left_prev1;

        PNode char_lstm_right_start = atomFeat.word_start >= 0 ? &(atomFeat.p_char_right_lstm->_hiddens[atomFeat.word_start]) : pseudo_char;
        PNode char_lstm_right_end = char_node_right_curr;

        char_span_repsent_left.forward(cg, char_lstm_left_end, char_lstm_left_start);
        char_span_repsent_right.forward(cg, char_lstm_right_start, char_lstm_right_end);

        //bichars
        PNode bichar_node_left_curr = (char_posi  < atomFeat.char_size - 1) ? &(atomFeat.p_bichar_left_lstm->_hiddens[char_posi]) : pseudo_bichar;  // i, i+1
        PNode bichar_node_left_next1 = (char_posi + 1  < atomFeat.char_size - 1) ? &(atomFeat.p_bichar_left_lstm->_hiddens[char_posi + 1]) : pseudo_bichar; // i+1, i+2
        PNode bichar_node_left_prev1 = (char_posi > 0 && char_posi  < atomFeat.char_size - 1) ? &(atomFeat.p_bichar_left_lstm->_hiddens[char_posi - 1]) : pseudo_bichar; // i-1, i
        PNode bichar_node_left_prev2 = (char_posi > 1) ? &(atomFeat.p_bichar_left_lstm->_hiddens[char_posi - 2]) : pseudo_bichar; // i-2, i-1

        PNode bichar_node_right_curr = (char_posi  < atomFeat.char_size - 1) ? &(atomFeat.p_bichar_right_lstm->_hiddens[char_posi]) : pseudo_bichar;   // i, i+1
        PNode bichar_node_right_next1 = (char_posi + 1  < atomFeat.char_size - 1) ? &(atomFeat.p_bichar_right_lstm->_hiddens[char_posi + 1]) : pseudo_bichar;  // i+1, i+2
        PNode bichar_node_right_prev1 = (char_posi > 0 && char_posi  < atomFeat.char_size - 1) ? &(atomFeat.p_bichar_right_lstm->_hiddens[char_posi - 1]) : pseudo_bichar; // i-1, i
        PNode bichar_node_right_prev2 = (char_posi > 1) ? &(atomFeat.p_bichar_right_lstm->_hiddens[char_posi - 2]) : pseudo_bichar; // i-2, i-1

        PNode bichar_lstm_left_start = (atomFeat.word_start > 1 && atomFeat.word_start < atomFeat.char_size - 1) ? &(atomFeat.p_bichar_left_lstm->_hiddens[atomFeat.word_start - 1]) : pseudo_bichar;
        PNode bichar_lstm_left_end = bichar_node_left_prev2;

        PNode bichar_lstm_right_start = (atomFeat.word_start >= 0 && atomFeat.word_start < atomFeat.char_size - 2) ? &(atomFeat.p_bichar_right_lstm->_hiddens[atomFeat.word_start]) : pseudo_bichar;
        PNode bichar_lstm_right_end = bichar_node_right_prev1;

        bichar_span_repsent_left.forward(cg, bichar_lstm_left_end, bichar_lstm_left_start);
        bichar_span_repsent_right.forward(cg, bichar_lstm_right_start, bichar_lstm_right_end);

        vector<PNode> word_components;
        last_word_input.forward(cg, atomFeat.str_word);
        word_components.push_back(&last_word_input);
        last_tag_input.forward(cg, atomFeat.str_tag);
        word_components.push_back(&last_tag_input);

        word_components.push_back(&char_span_repsent_left);
        word_components.push_back(&char_span_repsent_right);
        word_components.push_back(&bichar_span_repsent_left);
        word_components.push_back(&bichar_span_repsent_right);

        word_concat.forward(cg, word_components);
        word_represent.forward(cg, &word_concat);
        word_lstm.forward(cg, &word_represent, atomFeat.p_word_lstm);

        vector<PNode> wordNodes;
        wordNodes.push_back(&word_lstm._hidden);
        if (word_lstm._nSize > 1) {
            wordNodes.push_back(&word_lstm._pPrev->_hidden);
        } else {
            wordNodes.push_back(pseudo_word);
        }
        word_state_concat.forward(cg, wordNodes);
        word_state_hidden.forward(cg, &word_state_concat);


        //
        vector<PNode> char_components;
        char_components.push_back(char_node_left_curr);
        char_components.push_back(char_node_left_next1);
        char_components.push_back(char_node_left_next2);
        char_components.push_back(char_node_left_prev1);
        char_components.push_back(char_node_left_prev2);

        char_components.push_back(char_node_right_curr);
        char_components.push_back(char_node_right_next1);
        char_components.push_back(char_node_right_next2);
        char_components.push_back(char_node_right_prev1);
        char_components.push_back(char_node_right_prev2);

        char_components.push_back(&char_span_repsent_left);
        char_components.push_back(&char_span_repsent_right);

        //bichar
        char_components.push_back(bichar_node_left_curr);
        char_components.push_back(bichar_node_left_next1);
        char_components.push_back(bichar_node_left_prev1);
        char_components.push_back(bichar_node_left_prev2);

        char_components.push_back(bichar_node_right_curr);
        char_components.push_back(bichar_node_right_next1);
        char_components.push_back(bichar_node_right_prev1);
        char_components.push_back(bichar_node_right_prev2);

        char_components.push_back(&bichar_span_repsent_left);
        char_components.push_back(&bichar_span_repsent_right);

        char_state_concat.forward(cg, char_components);
        char_state_hidden.forward(cg, &char_state_concat);

        app_state_represent.forward(cg, &char_state_hidden);
        sep_state_represent.forward(cg, &char_state_hidden, &word_state_hidden);

        for (int idx = 0; idx < ac_num; idx++) {
            ac.set(actions[idx]);

            sumNodes.clear();

            string action_name = ac.str(opt);
            current_action_input[idx].forward(cg, action_name);
            if (ac.isAppend()) {
                action_score[idx].forward(cg, &current_action_input[idx], &app_state_represent);
            } else if (ac.isSeparate() || ac.isFinish()) {
                action_score[idx].forward(cg, &current_action_input[idx], &sep_state_represent);
            } else {
                std::cout << "error action here" << std::endl;
            }
            sumNodes.push_back(&action_score[idx]);

            if (prevStateNode != NULL) {
                sumNodes.push_back(prevStateNode);
            }

            outputs[idx].forward(cg, sumNodes);
        }
    }

};

#endif /* SRC_ActionedNodes_H_ */
