#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3LDG.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams {
    //required
    int beam;
    int maxlength;
    int action_num;
    dtype delta;
    int batch;


    dtype nnRegular; // for optimization
    dtype adaAlpha;  // for optimization
    dtype adaEps; // for optimization
    dtype dropProb;

    int char_dim;
    bool char_tune;
    int bichar_dim;
    bool bichar_tune;
    int chartype_dim; //must tune
    int char_concat_dim;

    int char_lstm_dim;
    int bichar_lstm_dim;

    int char_feat_dim;

    int char_state_dim;



    int word_dim; // not tune
    int tag_dim;  // must tune
    int word_concat_dim;
    int word_represent_dim;
    int word_lstm_dim;
    int word_feat_dim;
    int word_state_dim;

    int state_hidden_dim;

    Alphabet postags;

  public:
    HyperParams() {
        maxlength = max_sentence_clength + 1;
        bAssigned = false;
    }

  public:
    void setRequared(Options& opt) {
        //please specify dictionary outside
        //please sepcify char_dim, word_dim and action_dim outside.
        beam = opt.beam;
        delta = opt.delta;
        bAssigned = true;

        nnRegular = opt.regParameter;
        adaAlpha = opt.adaAlpha;
        adaEps = opt.adaEps;
        dropProb = opt.dropProb;
        batch = opt.batchSize;

        char_dim = opt.charEmbSize;
        char_tune = opt.charEmbFineTune;
        bichar_dim = opt.bicharEmbSize;
        bichar_tune = opt.bicharEmbFineTune;
        chartype_dim = opt.charTypeEmbSize;
        char_concat_dim = char_dim + chartype_dim;

        char_lstm_dim = opt.charRNNHiddenSize;
        bichar_lstm_dim = opt.charRNNHiddenSize + 30;

        char_feat_dim = 12 * char_lstm_dim + 10 * bichar_lstm_dim;
        char_state_dim = opt.charStateSize;

        word_dim = opt.wordEmbSize;
        tag_dim = opt.tagEmbSize;
        word_concat_dim = word_dim + tag_dim + 2 * char_lstm_dim + 2 * bichar_lstm_dim;
        word_represent_dim = opt.wordHiddenSize;
        word_lstm_dim = opt.wordRNNHiddenSize;
        word_feat_dim = 2 * word_lstm_dim;
        word_state_dim = opt.wordStateSize;

        state_hidden_dim = opt.stateHiddenSize;
    }

    void clear() {
        bAssigned = false;
    }

    bool bValid() {
        return bAssigned;
    }


  public:

    void print() {

    }

  private:
    bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */