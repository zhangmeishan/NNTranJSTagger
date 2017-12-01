#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams {
  public:
    Alphabet ext_embeded_chars; // chars
    LookupTable ext_char_table; // should be initialized outside
    Alphabet ext_embeded_bichars; // chars
    LookupTable ext_bichar_table; // should be initialized outside
    Alphabet ext_embeded_words; // words
    LookupTable ext_word_table; // should be initialized outside

    //neural parameters
    Alphabet embeded_chars; // chars
    LookupTable char_table; // should be initialized outside
    Alphabet embeded_bichars; // chars
    LookupTable bichar_table; // should be initialized outside

    Alphabet embeded_tags; // tags
    LookupTable tag_table; // should be initialized outside

    UniParams char_tanh_conv;
    LSTM1Params char_left_lstm; //left lstm
    LSTM1Params char_right_lstm; //right lstm

    UniParams word_represent;
    LSTM1Params word_lstm;

    UniParams state_represent;

    Alphabet embeded_actions;
    LookupTable scored_action_table;

  public:
    bool initial(HyperParams& opts) {
        // some model parameters should be initialized outside
        //neural features
        char_tanh_conv.initial(opts.char_hidden_dim, opts.char_concat_dim, true);
        char_left_lstm.initial(opts.char_lstm_dim, opts.char_hidden_dim); //left lstm
        char_right_lstm.initial(opts.char_lstm_dim, opts.char_hidden_dim); //right lstm

        word_represent.initial(opts.word_represent_dim, opts.word_concat_dim);
        word_lstm.initial(opts.word_lstm_dim, opts.word_represent_dim);

        state_represent.initial(opts.state_hidden_dim, opts.state_feat_dim, true);

        scored_action_table.initial(&embeded_actions, opts.state_hidden_dim, true);
        scored_action_table.E.val.random(0.01);

        return true;
    }


    void exportModelParams(ModelUpdate& ada) {
        char_table.exportAdaParams(ada);
        bichar_table.exportAdaParams(ada);
        tag_table.exportAdaParams(ada);

        char_tanh_conv.exportAdaParams(ada);
        char_left_lstm.exportAdaParams(ada);
        char_right_lstm.exportAdaParams(ada);

        word_represent.exportAdaParams(ada);
        word_lstm.exportAdaParams(ada);

        state_represent.exportAdaParams(ada);

        scored_action_table.exportAdaParams(ada);
    }

    // will add it later
    void saveModel() {

    }

    void loadModel(const string& inFile) {

    }

};

#endif /* SRC_ModelParams_H_ */