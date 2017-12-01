#ifndef _PARSER_OPTIONS_
#define _PARSER_OPTIONS_

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "N3LDG.h"

using namespace std;

class Options {
  public:
    int wordCutOff;
    int featCutOff;
    int charCutOff;
    int bicharCutOff;
    dtype initRange;
    int maxIter;
    int batchSize;
    dtype adaEps;
    dtype adaAlpha;
    dtype regParameter;
    dtype dropProb;
    dtype delta;
    dtype clip;
    dtype rpRatio;
    int beam;

    int wordStateSize;
    int charStateSize;
    int stateHiddenSize;
    int startBeam;

    int wordEmbSize;
    int lengthEmbSize;
    int wordNgram;
    int wordHiddenSize;
    int wordRNNHiddenSize;
    bool wordEmbFineTune;
    bool wordEmbNormalize;
    string wordEmbFile;

    int charEmbSize;
    int charTypeEmbSize;
    int bicharEmbSize;
    int charcontext;
    int charHiddenSize;
    int charRNNHiddenSize;
    bool charEmbFineTune;
    bool charEmbNormalize;
    bool bicharEmbFineTune;
    bool bicharEmbNormalize;
    string charEmbFile;
    string bicharEmbFile;

    string ngramEmbFile;
    string wordFile;

    int tagEmbSize;
    int ngramEmbSize;
    int tagStateSize;
    int tagRNNHiddenSize;

    int actionEmbSize;
    int actionNgram;
    int actionStateSize;
    int actionRNNHiddenSize;

    int verboseIter;
    bool saveIntermediate;
    bool train;
    int maxInstance;
    vector<string> testFiles;
    string outBest;
    int base;

    Options() {
        wordCutOff = 3;
        featCutOff = 0;
        charCutOff = 1;
        bicharCutOff = 1;
        initRange = 0.01;
        maxIter = 1000;
        batchSize = 1;
        adaEps = 1e-6;
        adaAlpha = 0.01;
        regParameter = 1e-8;
        dropProb = -1;
        delta = 0.0;
        clip = -1.0;
        rpRatio = 0.05;
        beam = 16;

        wordStateSize = 200;
        charStateSize = 200;
        stateHiddenSize = 200;
        startBeam = 1000;

        wordEmbSize = 200;
        lengthEmbSize = 20;
        wordNgram = 2;
        wordHiddenSize = 200;
        wordRNNHiddenSize = 200;
        wordEmbFineTune = true;
        wordEmbNormalize = false;
        wordEmbFile = "";

        charEmbSize = 200;
        charTypeEmbSize = 20;
        bicharEmbSize = 200;
        charcontext = 2;
        charHiddenSize = 200;
        charRNNHiddenSize = 200;
        charEmbFineTune = true;
        charEmbNormalize = false;
        bicharEmbFineTune = true;
        bicharEmbNormalize = false;
        charEmbFile = "";
        bicharEmbFile = "";

        ngramEmbFile = "";
        wordFile = "";

        tagEmbSize = 100;
        ngramEmbSize = 2;
        tagStateSize = 100;
        tagRNNHiddenSize = 100;

        actionEmbSize = 20;
        actionNgram = 2;
        actionStateSize = 30;
        actionRNNHiddenSize = 20;

        verboseIter = 100;
        saveIntermediate = true;
        train = false;
        maxInstance = -1;
        testFiles.clear();
        outBest = "";
        base = 1;
    }

    virtual ~Options() {

    }

    void setOptions(const vector<string> &vecOption) {
        int i = 0;
        for (; i < vecOption.size(); ++i) {
            pair<string, string> pr;
            string2pair(vecOption[i], pr, '=');
            if (pr.first == "wordCutOff")
                wordCutOff = atoi(pr.second.c_str());
            if (pr.first == "featCutOff")
                featCutOff = atoi(pr.second.c_str());
            if (pr.first == "charCutOff")
                charCutOff = atoi(pr.second.c_str());
            if (pr.first == "bicharCutOff")
                bicharCutOff = atoi(pr.second.c_str());
            if (pr.first == "initRange")
                initRange = atof(pr.second.c_str());
            if (pr.first == "maxIter")
                maxIter = atoi(pr.second.c_str());
            if (pr.first == "batchSize")
                batchSize = atoi(pr.second.c_str());
            if (pr.first == "adaEps")
                adaEps = atof(pr.second.c_str());
            if (pr.first == "adaAlpha")
                adaAlpha = atof(pr.second.c_str());
            if (pr.first == "regParameter")
                regParameter = atof(pr.second.c_str());
            if (pr.first == "dropProb")
                dropProb = atof(pr.second.c_str());
            if (pr.first == "delta")
                delta = atof(pr.second.c_str());
            if (pr.first == "clip")
                clip = atof(pr.second.c_str());
            if (pr.first == "rpRatio")
                rpRatio = atof(pr.second.c_str());
            if (pr.first == "beam")
                beam = atoi(pr.second.c_str());

            if (pr.first == "wordStateSize")
                wordStateSize = atoi(pr.second.c_str());
            if (pr.first == "charStateSize")
                charStateSize = atoi(pr.second.c_str());
            if (pr.first == "stateHiddenSize")
                stateHiddenSize = atoi(pr.second.c_str());
            if (pr.first == "startBeam")
                startBeam = atoi(pr.second.c_str());

            if (pr.first == "wordEmbSize")
                wordEmbSize = atoi(pr.second.c_str());
            if (pr.first == "lengthEmbSize")
                lengthEmbSize = atoi(pr.second.c_str());
            if (pr.first == "wordNgram")
                wordNgram = atoi(pr.second.c_str());
            if (pr.first == "wordHiddenSize")
                wordHiddenSize = atoi(pr.second.c_str());
            if (pr.first == "wordRNNHiddenSize")
                wordRNNHiddenSize = atoi(pr.second.c_str());
            if (pr.first == "wordEmbFineTune")
                wordEmbFineTune = (pr.second == "true") ? true : false;
            if (pr.first == "wordEmbNormalize")
                wordEmbNormalize = (pr.second == "true") ? true : false;
            if (pr.first == "wordEmbFile")
                wordEmbFile = pr.second;

            if (pr.first == "charEmbSize")
                charEmbSize = atoi(pr.second.c_str());
            if (pr.first == "charTypeEmbSize")
                charTypeEmbSize = atoi(pr.second.c_str());
            if (pr.first == "bicharEmbSize")
                bicharEmbSize = atoi(pr.second.c_str());
            if (pr.first == "charcontext")
                charcontext = atoi(pr.second.c_str());
            if (pr.first == "charHiddenSize")
                charHiddenSize = atoi(pr.second.c_str());
            if (pr.first == "charRNNHiddenSize")
                charRNNHiddenSize = atoi(pr.second.c_str());
            if (pr.first == "charEmbFineTune")
                charEmbFineTune = (pr.second == "true") ? true : false;
            if (pr.first == "charEmbNormalize")
                charEmbNormalize = (pr.second == "true") ? true : false;
            if (pr.first == "bicharEmbFineTune")
                bicharEmbFineTune = (pr.second == "true") ? true : false;
            if (pr.first == "bicharEmbNormalize")
                bicharEmbNormalize = (pr.second == "true") ? true : false;
            if (pr.first == "charEmbFile")
                charEmbFile = pr.second;
            if (pr.first == "bicharEmbFile")
                bicharEmbFile = pr.second;

            if (pr.first == "ngramEmbFile")
                ngramEmbFile = pr.second;
            if (pr.first == "wordFile")
                wordFile = pr.second;

            if (pr.first == "tagEmbSize")
                tagEmbSize = atoi(pr.second.c_str());
            if (pr.first == "ngramEmbSize")
                ngramEmbSize = atoi(pr.second.c_str());
            if (pr.first == "tagStateSize")
                tagStateSize = atoi(pr.second.c_str());
            if (pr.first == "tagRNNHiddenSize")
                tagRNNHiddenSize = atoi(pr.second.c_str());

            if (pr.first == "actionEmbSize")
                actionEmbSize = atoi(pr.second.c_str());
            if (pr.first == "actionNgram")
                actionNgram = atoi(pr.second.c_str());
            if (pr.first == "actionStateSize")
                actionStateSize = atoi(pr.second.c_str());
            if (pr.first == "actionRNNHiddenSize")
                actionRNNHiddenSize = atoi(pr.second.c_str());

            if (pr.first == "verboseIter")
                verboseIter = atoi(pr.second.c_str());
            if (pr.first == "train")
                train = (pr.second == "true") ? true : false;
            if (pr.first == "saveIntermediate")
                saveIntermediate = (pr.second == "true") ? true : false;
            if (pr.first == "maxInstance")
                maxInstance = atoi(pr.second.c_str());
            if (pr.first == "testFile")
                testFiles.push_back(pr.second);
            if (pr.first == "outBest")
                outBest = pr.second;
            if (pr.first == "base")
                base = atoi(pr.second.c_str());

        }
    }

    void showOptions() {
        std::cout << std::endl;
        std::cout << "show options" << std::endl;
        std::cout << "wordCutOff = " << wordCutOff << std::endl;
        std::cout << "featCutOff = " << featCutOff << std::endl;
        std::cout << "charCutOff = " << charCutOff << std::endl;
        std::cout << "bicharCutOff = " << bicharCutOff << std::endl;
        std::cout << "initRange = " << initRange << std::endl;
        std::cout << "maxIter = " << maxIter << std::endl;
        std::cout << "batchSize = " << batchSize << std::endl;
        std::cout << "adaEps = " << adaEps << std::endl;
        std::cout << "adaAlpha = " << adaAlpha << std::endl;
        std::cout << "regParameter = " << regParameter << std::endl;
        std::cout << "dropProb = " << dropProb << std::endl;
        std::cout << "delta = " << delta << std::endl;
        std::cout << "clip = " << clip << std::endl;
        std::cout << "rpRatio = " << rpRatio << std::endl;
        std::cout << "beam = " << beam << std::endl;

        std::cout << "wordStateSize = " << wordStateSize << std::endl;
        std::cout << "charStateSize = " << charStateSize << std::endl;
        std::cout << "stateHiddenSize = " << stateHiddenSize << std::endl;
        std::cout << "startBeam = " << startBeam << std::endl;

        std::cout << "wordEmbSize = " << wordEmbSize << std::endl;
        std::cout << "lengthEmbSize = " << lengthEmbSize << std::endl;
        std::cout << "wordNgram = " << wordNgram << std::endl;
        std::cout << "wordHiddenSize = " << wordHiddenSize << std::endl;
        std::cout << "wordRNNHiddenSize = " << wordRNNHiddenSize << std::endl;
        std::cout << "wordEmbFineTune = " << wordEmbFineTune << std::endl;
        std::cout << "wordEmbNormalize = " << wordEmbNormalize << std::endl;
        std::cout << "wordEmbFile = " << wordEmbFile << std::endl;

        std::cout << "charEmbSize = " << charEmbSize << std::endl;
        std::cout << "charTypeEmbSize = " << charEmbSize << std::endl;
        std::cout << "bicharEmbSize = " << bicharEmbSize << std::endl;
        std::cout << "charcontext = " << charcontext << std::endl;
        std::cout << "charHiddenSize = " << charHiddenSize << std::endl;
        std::cout << "charRNNHiddenSize = " << charRNNHiddenSize << std::endl;
        std::cout << "charEmbFineTune = " << charEmbFineTune << std::endl;
        std::cout << "charEmbNormalize = " << charEmbNormalize << std::endl;
        std::cout << "bicharEmbFineTune = " << bicharEmbFineTune << std::endl;
        std::cout << "bicharEmbNormalize = " << bicharEmbNormalize << std::endl;
        std::cout << "charEmbFile = " << charEmbFile << std::endl;
        std::cout << "bicharEmbFile = " << bicharEmbFile << std::endl;

        std::cout << "ngramEmbFile = " << ngramEmbFile << std::endl;
        std::cout << "wordFile = " << wordFile << std::endl;

        std::cout << "tagEmbSize = " << tagEmbSize << std::endl;
        std::cout << "ngramEmbSize = " << ngramEmbSize << std::endl;
        std::cout << "tagStateSize = " << tagStateSize << std::endl;
        std::cout << "tagRNNHiddenSize = " << tagRNNHiddenSize << std::endl;

        std::cout << "actionEmbSize = " << actionEmbSize << std::endl;
        std::cout << "actionNgram = " << actionNgram << std::endl;
        std::cout << "actionStateSize = " << actionStateSize << std::endl;
        std::cout << "actionRNNHiddenSize = " << actionRNNHiddenSize << std::endl;

        std::cout << "verboseIter = " << verboseIter << std::endl;
        std::cout << "saveItermediate = " << saveIntermediate << std::endl;
        std::cout << "train = " << train << std::endl;
        std::cout << "maxInstance = " << maxInstance << std::endl;
        for (int idx = 0; idx < testFiles.size(); idx++) {
            std::cout << "testFile = " << testFiles[idx] << std::endl;
        }
        std::cout << "outBest = " << outBest << std::endl;
        std::cout << "base = " << base << std::endl;

        std::cout << std::endl;
    }

    void load(const std::string& infile) {
        ifstream inf;
        inf.open(infile.c_str());
        vector<string> vecLine;
        while (1) {
            string strLine;
            if (!my_getline(inf, strLine)) {
                break;
            }
            if (strLine.empty())
                continue;
            vecLine.push_back(strLine);
        }
        inf.close();
        setOptions(vecLine);
    }
};

#endif

