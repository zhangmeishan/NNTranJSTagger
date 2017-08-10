#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams {
  public:
    //neural parameters
    Alphabet embeded_chars; // chars
    LookupTable char_table; // should be initialized outside
    Alphabet embeded_bichars; // chars
    LookupTable bichar_table; // should be initialized outside
    Alphabet embeded_chartypes; // chars
    LookupTable chartype_table; // should be initialized outside

    Alphabet embeded_ngram_chars; // ngram chars
    LookupTable ngram_char_table; // should be initialized outside
    Alphabet embeded_tags; // tags
    LookupTable tag_table; // should be initialized outside

    LSTM1Params char_left_lstm; //left lstm
    LSTM1Params char_right_lstm; //right lstm
    LSTM1Params bichar_left_lstm; //left lstm
    LSTM1Params bichar_right_lstm; //right lstm

    UniParams word_represent;
    LSTM1Params word_lstm;
    UniParams word_state_hidden;

    UniParams char_state_hidden;

    UniParams app_state_represent;
    BiParams sep_state_represent;

    Alphabet embeded_actions;
    LookupTable scored_action_table;

  public:
    bool initial(HyperParams& opts) {
        // some model parameters should be initialized outside
        //neural features
        char_left_lstm.initial(opts.char_lstm_dim, opts.char_concat_dim); //left lstm
        char_right_lstm.initial(opts.char_lstm_dim, opts.char_concat_dim); //right lstm
        bichar_left_lstm.initial(opts.bichar_lstm_dim, opts.bichar_dim); //left lstm
        bichar_right_lstm.initial(opts.bichar_lstm_dim, opts.bichar_dim); //right lstm

        word_represent.initial(opts.word_represent_dim, opts.word_concat_dim);
        word_lstm.initial(opts.word_lstm_dim, opts.word_represent_dim);
        word_state_hidden.initial(opts.word_state_dim, opts.word_feat_dim, true);

        char_state_hidden.initial(opts.char_state_dim, opts.char_feat_dim, true);

        app_state_represent.initial(opts.state_hidden_dim, opts.char_state_dim, true);
        sep_state_represent.initial(opts.state_hidden_dim, opts.char_state_dim, opts.word_state_dim, true);

        scored_action_table.initial(&embeded_actions, opts.state_hidden_dim, true);
        scored_action_table.E.val.random(0.01);

        return true;
    }


    void exportModelParams(ModelUpdate& ada) {
        char_table.exportAdaParams(ada);
        chartype_table.exportAdaParams(ada);
        bichar_table.exportAdaParams(ada);
        //ngram_char_table.exportAdaParams(ada); // no tune
        tag_table.exportAdaParams(ada);

        char_left_lstm.exportAdaParams(ada);
        char_right_lstm.exportAdaParams(ada);
        bichar_left_lstm.exportAdaParams(ada);
        bichar_right_lstm.exportAdaParams(ada);

        word_represent.exportAdaParams(ada);
        word_lstm.exportAdaParams(ada);
        word_state_hidden.exportAdaParams(ada);

        char_state_hidden.exportAdaParams(ada);

        app_state_represent.exportAdaParams(ada);
        sep_state_represent.exportAdaParams(ada);

        scored_action_table.exportAdaParams(ada);
    }


    // will add it later
    void saveModel() {

    }

    void loadModel(const string& inFile) {

    }

};

#endif /* SRC_ModelParams_H_ */