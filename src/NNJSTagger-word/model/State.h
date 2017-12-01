/*
 * State.h
 *
 *  Created on: Oct 1, 2015
 *      Author: mszhang
 */

#ifndef SEG_STATE_H_
#define SEG_STATE_H_

#include "ModelParams.h"
#include "Action.h"
#include "ActionedNodes.h"
#include "AtomFeatures.h"
#include "Utf.h"
#include "GlobalNodes.h"
#include  "Instance.h"

class CStateItem {
  public:
    std::string _word;
    int _tag;
    int _wstart;
    int _wend;
    CStateItem *_prevStackState;
    CStateItem *_prevState;
    int _next_index;

    const std::vector<std::string> *_chars;
    int _char_size;
    int _word_count;

    CAction _lastAction;
    PNode _score;

    // features
    ActionedNodes _nextscores;  // features current used
    AtomFeatures _atomFeat;  //features will be used for future

  public:
    bool _bStart; // whether it is a start state
    bool _bGold; // for train

  public:
    CStateItem() {
        clear();
    }


    virtual ~CStateItem() {
        clear();
    }

    void initial(ModelParams& params, HyperParams& hyparams) {
        _nextscores.initial(params, hyparams);
    }

    void setInput(const std::vector<std::string>* pCharacters) {
        _chars = pCharacters;
        _char_size = pCharacters->size();
    }

    void clear() {
        _word = "</s>";
        _tag = -1;
        _wstart = -1;
        _wend = -1;
        _prevStackState = 0;
        _prevState = 0;
        _next_index = 0;
        _chars = 0;
        _char_size = 0;
        _lastAction.clear();
        _word_count = 0;
        _bStart = true;
        _bGold = true;
        _score = NULL;
    }


    const CStateItem* getPrevStackState() const {
        return _prevStackState;
    }

    const CStateItem* getPrevState() const {
        return _prevState;
    }

    std::string getLastWord() {
        return _word;
    }


  public:
    //only assign context
    void separate(CStateItem* next, int tag) {
        if (_next_index >= _char_size) {
            std::cout << "separate error" << std::endl;
            return;
        }
        next->_word = (*_chars)[_next_index];
        next->_tag = tag;
        next->_wstart = _next_index;
        next->_wend = _next_index;
        next->_prevStackState = this;
        next->_prevState = this;
        next->_next_index = _next_index + 1;
        next->_chars = _chars;
        next->_char_size = _char_size;
        next->_word_count = _word_count + 1;
        next->_lastAction.set(CAction::SEP, tag);
    }

    //only assign context
    void finish(CStateItem* next) {
        if (_next_index != _char_size) {
            std::cout << "finish error" << std::endl;
            return;
        }
        next->_word = "</s>";
        next->_tag = -1;
        next->_wstart = -1;
        next->_wend = -1;
        next->_prevStackState = this;
        next->_prevState = this;
        next->_next_index = _next_index + 1;
        next->_chars = _chars;
        next->_char_size = _char_size;
        next->_word_count = _word_count;
        next->_lastAction.set(CAction::FIN, -1);
    }

    //only assign context
    void append(CStateItem* next) {
        if (_next_index >= _char_size) {
            std::cout << "append error" << std::endl;
            return;
        }
        next->_word = _word + (*_chars)[_next_index];
        next->_tag = _tag;
        next->_wstart = _wstart;
        next->_wend = _next_index;
        next->_prevStackState = _prevStackState;
        next->_prevState = this;
        next->_next_index = _next_index + 1;
        next->_chars = _chars;
        next->_char_size = _char_size;
        next->_word_count = _word_count;
        next->_lastAction.set(CAction::APP, -1);
    }

    void move(CStateItem* next, const CAction& ac) {
        if (ac.isAppend()) {
            append(next);
        } else if (ac.isSeparate()) {
            separate(next, ac._tag);
        } else if (ac.isFinish()) {
            finish(next);
        } else {
            std::cout << "error action" << std::endl;
        }

        next->_bStart = false;
        next->_bGold = false;
    }

    bool IsTerminated() const {
        if (_lastAction.isFinish())
            return true;
        return false;
    }

    //partial results
    void getResults(std::vector<std::string>& words, std::vector<std::string>& tags, HyperParams* opt) const {
        words.clear();
        tags.clear();
        vector<const CStateItem *> preSepStates;
        preSepStates.clear();
        if (!IsTerminated()) {
            preSepStates.insert(preSepStates.begin(), this);
        }
        const CStateItem *prevStackState = _prevStackState;
        while (prevStackState != 0 && !prevStackState->_bStart) {
            preSepStates.insert(preSepStates.begin(), prevStackState);
            prevStackState = prevStackState->_prevStackState;
        }
        //will add results
        int state_num;
        state_num = preSepStates.size();
        if (state_num != _word_count) {
            std::cout << "bug exists: " << state_num << " " << _word_count << std::endl;
        }
        for (int idx = 0; idx < state_num; idx++) {
            words.push_back(preSepStates[idx]->_word);
            tags.push_back(opt->postags.getTagName(preSepStates[idx]->_tag));
        }
    }


    void getGoldAction(const Instance& inst, CAction& ac, HyperParams* opt) const {
        if (_next_index == _char_size) {
            ac.set(CAction::FIN, -1);
            return;
        }
        if (_next_index == 0) {
            ac.set(CAction::SEP, opt->postags.getTagId(inst.tags[_word_count]));
            return;
        }

        if (_next_index > 0 && _next_index < _char_size) {
            // should have a check here to see whether the words are match, but I did not do it here
            if (_word.length() == inst.words[_word_count - 1].length()) {
                ac.set(CAction::SEP, opt->postags.getTagId(inst.tags[_word_count]));
                return;
            } else {
                ac.set(CAction::APP, -1);
                return;
            }
        }

        ac.set(CAction::NO_ACTION, -1);
        return;
    }

    // we did not judge whether history actions are match with current state.
    void getGoldAction(const CStateItem* goldState, CAction& ac) const {
        if (_next_index > goldState->_next_index || _next_index < 0) {
            ac.set(CAction::NO_ACTION, -1);
            return;
        }
        const CStateItem *prevState = goldState->_prevState;
        CAction curAction = goldState->_lastAction;
        while (_next_index < prevState->_next_index) {
            curAction = prevState->_lastAction;
            prevState = prevState->_prevState;
        }
        return ac.set(curAction);
    }

    void getCandidateActions(vector<CAction> & actions, HyperParams* opt) const {
        actions.clear();
        CAction ac;
        if (_next_index == 0) {
            for (int idx = 0; idx < opt->postags.size(); idx++) {
                if (opt->postags.canAssignCharTag((*_chars)[_next_index], idx)) {
                    ac.set(CAction::SEP, idx);
                    actions.push_back(ac);
                }
            }
        } else if (_next_index == _char_size) {
            bool lastTagValid = opt->postags.canAssignWordTag(_word, _tag);
            dtype delta = lastTagValid ? 0 : opt->delta;
            ac.set(CAction::FIN, -1, delta);
            actions.push_back(ac);

        } else if (_next_index > 0 && _next_index < _char_size) {
            bool lastTagValid = opt->postags.canAssignWordTag(_word, _tag);
            dtype delta = lastTagValid ? 0 : opt->delta;
            for (int idx = 0; idx < opt->postags.size(); idx++) {
                if (opt->postags.canAssignCharTag((*_chars)[_next_index], idx)) {
                    ac.set(CAction::SEP, idx, delta);
                    actions.push_back(ac);
                }
            }
            ac.set(CAction::APP, -1);
            actions.push_back(ac);
        } else {

        }

    }

    inline std::string str(HyperParams* opt) const {
        stringstream curoutstr;
        curoutstr << "score: " << _score->val[0] << " ";
        curoutstr << "seg:";
        std::vector<std::string> words;
        std::vector<std::string> tags;
        getResults(words, tags, opt);
        for (int idx = 0; idx < words.size(); idx++) {
            curoutstr << " " << words[idx];
        }

        return curoutstr.str();
    }


  public:
    //bGready always true here
    inline void computeNextScore(Graph *cg, const vector<CAction>& acs, bool bGreedy) {
        if (_bStart || bGreedy) {
            _nextscores.forward(cg, acs, _atomFeat, NULL);
        } else {
            _nextscores.forward(cg, acs, _atomFeat, _score);
        }
    }

    inline void prepare(HyperParams* hyper_params, ModelParams* model_params, GlobalNodes* global_nodes) {
        _atomFeat.str_word = _word;
        _atomFeat.str_tag = hyper_params->postags.getTagName(_tag);
        int word_length = _next_index - _wstart;
        if (word_length > 6) {
            word_length = 6;
        }
        if (word_length < 0) {
            word_length = 0;
        }
        stringstream ss_word_length;
        ss_word_length << word_length;
        _atomFeat.str_len = ss_word_length.str();
        _atomFeat.word_num = _word_count;
        _atomFeat.word_start = _wstart;
        _atomFeat.next_position = _next_index;
        _atomFeat.char_size = _char_size;
        _atomFeat.p_word_lstm = _prevStackState == 0 ? NULL : &(_prevStackState->_nextscores.word_lstm);
        _atomFeat.p_char_left_lstm = global_nodes == NULL ? NULL : &(global_nodes->char_left_lstm);
        _atomFeat.p_char_right_lstm = global_nodes == NULL ? NULL : &(global_nodes->char_right_lstm);
    }
};


class CScoredState {
  public:
    CStateItem *item;
    CAction ac;
    dtype score;
    bool bGold;
    int position;

  public:
    CScoredState() : item(0), score(0), ac(), bGold(0), position(-1) {
    }

    CScoredState(const CScoredState& other) : item(other.item), score(other.score), ac(other.ac), bGold(other.bGold), position(other.position) {

    }

  public:
    bool operator <(const CScoredState &a1) const {
        return score < a1.score;
    }
    bool operator >(const CScoredState &a1) const {
        return score > a1.score;
    }
    bool operator <=(const CScoredState &a1) const {
        return score <= a1.score;
    }
    bool operator >=(const CScoredState &a1) const {
        return score >= a1.score;
    }
};

class CScoredState_Compare {
  public:
    int operator()(const CScoredState &o1, const CScoredState &o2) const {

        if (o1.score < o2.score)
            return -1;
        else if (o1.score > o2.score)
            return 1;
        else
            return 0;
    }
};


struct COutput {
    PNode in;
    bool bGold;

    COutput() : in(NULL), bGold(0) {
    }

    COutput(const COutput& other) : in(other.in), bGold(other.bGold) {
    }
};



#endif /* SEG_STATE_H_ */
