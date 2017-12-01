/*
 * JSTagger.h
 *
 *  Created on: Jan 25, 2016
 *      Author: mszhang
 */

#ifndef SRC_PARSER_H_
#define SRC_PARSER_H_

#include "N3LDG.h"
#include "Driver.h"
#include "Options.h"
#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;


class JSTagger {
  public:
    JSTagger();
    virtual ~JSTagger();

  public:
    Driver m_driver;
    Options m_options;
    Pipe m_pipe;

  public:
    int createAlphabet(const vector<Instance>& vecInsts);

  public:
    void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile);
    void predict(const vector<Instance>& inputs, vector<vector<string> >& seg_results, vector<vector<string> >& tag_results);
    void test(const string& testFile, const string& outputFile, const string& modelFile);
    void getGoldActions(const vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions);

  public:
    void writeModelFile(const string& outputModelFile);
    void loadModelFile(const string& inputModelFile);

};

#endif /* SRC_PARSER_H_ */
