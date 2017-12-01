#ifndef SRC_GreedyGraph_H_
#define SRC_GreedyGraph_H_

#include "ModelParams.h"
#include "State.h"

// Each model consists of two parts, building neural graph and defining output losses.
// This framework wastes memory
class GreedyGraphBuilder {
  public:
    GlobalNodes globalNodes;
    // node instances
    CStateItem start;
    vector<CStateItem> states;
    vector<vector<COutput> > outputs;

  private:
    ModelParams *pModel;
    HyperParams *pOpts;

    // node pointers
  public:
    GreedyGraphBuilder() {
        clear();
    }

    ~GreedyGraphBuilder() {
        clear();
    }

  public:
    //allocate enough nodes
    inline void initial(ModelParams &model, HyperParams &opts) {
        std::cout << "state size: " << sizeof(CStateItem) << std::endl;
        std::cout << "action node size: " << sizeof(ActionedNodes) << std::endl;
        globalNodes.resize(max_sentence_clength);
        states.resize(opts.maxlength + 1);

        globalNodes.initial(model, opts);
        for (int idx = 0; idx < states.size(); idx++) {
            states[idx].initial(model, opts);
        }
        start.clear();
        start.initial(model, opts);

        pModel = &model;
        pOpts = &opts;
    }

    inline void clear() {
        //beams.clear();
        clearVec(outputs);
        states.clear();
        pModel = NULL;
        pOpts = NULL;
    }

  public:
    inline void encode(Graph* pcg, const std::vector<std::string>* pCharacters) {
        globalNodes.forward(pcg, pCharacters);
    }

  public:
    // some nodes may behave different during training and decode, for example, dropout
    inline void decode(Graph* pcg, const std::vector<std::string>* pCharacters, const vector<CAction>* goldAC = NULL) {
        //first step, clear node values
        clearVec(outputs);

        //second step, build graph
        CStateItem* pGenerator;
        int step, offset;
        vector<CAction> actions; // actions to apply for a candidate
        CScoredState scored_action; // used rank actions
        COutput output;
        bool correct_action_scored;
        bool correct_in_beam;
        CAction answer, action;
        vector<COutput> per_step_output;
        NRHeap<CScoredState, CScoredState_Compare> beam;
        beam.resize(pOpts->action_num + 1);

        start.setInput(pCharacters);
        pGenerator = &start;

        step = 0;
        while (true) {
            //prepare for the next
            pGenerator->prepare(pOpts, pModel, &globalNodes);
            pGenerator->getCandidateActions(actions, pOpts);
            pGenerator->computeNextScore(pcg, actions, true);
            pcg->compute();

            answer.clear();
            per_step_output.clear();
            correct_action_scored = false;
            if (pcg->train) answer = (*goldAC)[step];
            beam.clear();
            scored_action.item = pGenerator;
            for (int idy = 0; idy < actions.size(); ++idy) {
                scored_action.ac.set(actions[idy]); //TODO:
                if (pGenerator->_bGold && actions[idy] == answer) {
                    scored_action.bGold = true;
                    correct_action_scored = true;
                    output.bGold = true;
                } else {
                    scored_action.bGold = false;
                    output.bGold = false;
                }
                if (pcg->train)pGenerator->_nextscores.outputs[idy].val[0] += pOpts->delta;
                scored_action.score = pGenerator->_nextscores.outputs[idy].val[0];
                scored_action.position = idy;
                output.in = &(pGenerator->_nextscores.outputs[idy]);
                beam.add_elem(scored_action);
                per_step_output.push_back(output);
            }

            outputs.push_back(per_step_output);

            // FIXME:
            if (pcg->train && !correct_action_scored) { //training
                std::cout << "error during training, gold-standard action is filtered: " << step << std::endl;
                std::cout << answer.str(pOpts) << std::endl;
                return;
            }

            offset = beam.elemsize();
            if (offset == 0) { // judge correctiveness
                std::cout << "error, reach no output here, please find why" << std::endl;
                for (int idx = 0; idx < pCharacters->size(); idx++) {
                    std::cout << (*pCharacters)[idx] << std::endl;
                }
                std::cout << "" << std::endl;
                return;
            }

            beam.sort_elem();
            if (pcg->train) {
                bool find_next = false;
                for (int idx = 0; idx < offset; idx++) {
                    if (beam[idx].bGold) {
                        pGenerator = beam[idx].item;
                        action.set(beam[idx].ac);
                        pGenerator->move(&(states[step]), action);
                        states[step]._bGold = beam[idx].bGold;
                        states[step]._score = &(pGenerator->_nextscores.outputs[beam[idx].position]);
                        find_next = true;
                    }
                }

                if (!find_next) {
                    std::cout << "serious bug here" << std::endl;
                    exit(0);
                }
            } else {
                pGenerator = beam[0].item;
                action.set(beam[0].ac);
                pGenerator->move(&(states[step]), action);
                states[step]._bGold = beam[0].bGold;
                states[step]._score = &(pGenerator->_nextscores.outputs[beam[0].position]);
            }
            pGenerator = &(states[step]);
            if (states[step].IsTerminated()) {
                break;
            }

            step++;
        }

        return;
    }

};

#endif /* SRC_GreedyGraph_H_ */