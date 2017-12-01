#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3LDG.h"
#include "Options.h"
#include "CTag.h"

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
    int char_concat_dim;
    int char_hidden_dim;
    int char_lstm_dim;



    int tag_dim;  // must tune
    int word_dim;
    int word_concat_dim;
    int word_represent_dim;
    int word_lstm_dim;


    int state_feat_dim;
    int state_hidden_dim;

    CTag postags;

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
        char_concat_dim = char_dim + bichar_dim + char_dim + bichar_dim;
        char_hidden_dim = opt.charHiddenSize;
        char_lstm_dim = opt.charRNNHiddenSize;


        word_dim = opt.wordEmbSize;
        tag_dim = opt.tagEmbSize;
        word_concat_dim = tag_dim + word_dim + 2 * char_lstm_dim;
        word_represent_dim = opt.wordHiddenSize;
        word_lstm_dim = opt.wordRNNHiddenSize;

        state_feat_dim = 2 * char_lstm_dim + word_lstm_dim;
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
        std::cout << "show hyper parameters" << std::endl;
        std::cout << "maxlength = " << maxlength << std::endl;
        std::cout << "beam = " << beam << std::endl;
        std::cout << "delta = " << delta << std::endl;
        std::cout << "action_num = " << action_num << std::endl;
        std::cout << "nnRegular = " << nnRegular << std::endl;
        std::cout << "adaAlpha = " << adaAlpha << std::endl;
        std::cout << "adaEps = " << adaEps << std::endl;
        std::cout << "dropProb = " << dropProb << std::endl;
        std::cout << "batch = " << batch << std::endl;

        std::cout << "char_dim = " << char_dim << std::endl;
        std::cout << "char_tune = " << char_tune << std::endl;
        std::cout << "bichar_dim = " << bichar_dim << std::endl;
        std::cout << "bichar_tune = " << bichar_tune << std::endl;
        std::cout << "char_concat_dim = " << char_concat_dim << std::endl;
        std::cout << "char_hidden_dim = " << char_hidden_dim << std::endl;
        std::cout << "char_lstm_dim = " << char_lstm_dim << std::endl;

        std::cout << "word_dim = " << word_dim << std::endl;
        std::cout << "tag_dim = " << tag_dim << std::endl;
        std::cout << "word_concat_dim = " << word_concat_dim << std::endl;
        std::cout << "word_represent_dim = " << word_represent_dim << std::endl;
        std::cout << "word_lstm_dim = " << word_lstm_dim << std::endl;

        std::cout << "state_feat_dim = " << state_feat_dim << std::endl;
        std::cout << "state_hidden_dim = " << state_hidden_dim << std::endl;

        std::cout << std::endl;
    }

  private:
    bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */