/*
 * JSTagger.cpp
 *
 *  Created on: Jan 25, 2016
 *      Author: mszhang
 */

#include "NNJSTagger.h"
#include <chrono>
#include <omp.h>
#include "Argument_helper.h"

JSTagger::JSTagger(){
    // TODO Auto-generated constructor stub
    srand(0);
    //Node::id = 0;
}

JSTagger::~JSTagger() {
    // TODO Auto-generated destructor stub
}

// all linear features are extracted from positive examples
int JSTagger::createAlphabet(const vector<Instance>& vecInsts) {
    cout << "Creating Alphabet..." << endl;

    int numInstance = vecInsts.size();

    unordered_map<string, int> char_stat;
    unordered_map<string, int> bichar_stat;
    
    assert(numInstance > 0);
    int count = 0;
    for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
        const Instance &instance = vecInsts[numInstance];
		int char_size = instance.charsize();
        for (int idx = 0; idx < char_size; idx++) {
            char_stat[instance.chars[idx]]++;
            if (idx < instance.charsize() - 1) {
                bichar_stat[instance.chars[idx] + instance.chars[idx + 1]]++;
            }
        }
		bichar_stat[instance.chars[char_size-1] + nullkey]++;
		bichar_stat[nullkey + instance.chars[0]]++;
        count += instance.wordsize();

		//initialize m_driver._hyperparams.postags
		for (int idx = 0; idx < instance.wordsize(); idx++) {
			string curTag = instance.tags[idx];
			int curTagId = m_driver._hyperparams.postags.getTagId(curTag);
			if (curTagId == -1) {
				std::cout << curTag << "is not corverd in predefined tag sets." << std::endl;
				continue;
			}
			string curWord = instance.words[idx];
			vector<string> charInfo;
			getCharactersFromUTF8String(curWord, charInfo);
			string firstChar = charInfo[0];

			m_driver._hyperparams.postags.firstchar_pos[firstChar].insert(curTagId);
			m_driver._hyperparams.postags.word_pos[curWord].insert(curTagId);
			m_driver._hyperparams.postags.char_freq[firstChar]++;
			m_driver._hyperparams.postags.word_freq[curWord]++;

			if (m_driver._hyperparams.postags.char_freq[firstChar] > m_driver._hyperparams.postags.maxFreqChar) {
				m_driver._hyperparams.postags.maxFreqChar = m_driver._hyperparams.postags.char_freq[firstChar];
			}

			if (m_driver._hyperparams.postags.word_freq[curWord] > m_driver._hyperparams.postags.maxFreqWord) {
				m_driver._hyperparams.postags.maxFreqWord = m_driver._hyperparams.postags.word_freq[curWord];
			}
		}
    }

	m_driver._hyperparams.action_num = m_driver._hyperparams.postags.size() + 2;
    char_stat[nullkey] = m_options.charCutOff + 1;
    char_stat[unknownkey] = m_options.charCutOff + 1;
    bichar_stat[nullkey] = m_options.bicharCutOff + 1;
    bichar_stat[unknownkey] = m_options.bicharCutOff + 1;

	//char
    if (m_options.charEmbFile != "") {
        m_driver._modelparams.ext_embeded_chars.initial(m_options.charEmbFile);
		std::cout << "Embedding char file vocabulary size: " << m_driver._modelparams.ext_embeded_chars.size() << std::endl;
    }
	else {
		std::cout << "char embedding file not specified." << std::endl;
		exit(0);
	}
    m_driver._modelparams.embeded_chars.initial(char_stat, m_options.charCutOff);
	std::cout << "fine tuned char vocabulary size: " << m_driver._modelparams.embeded_chars.size() << std::endl;

	//bichar
	if (m_options.bicharEmbFile != "") {
		m_driver._modelparams.ext_embeded_bichars.initial(m_options.bicharEmbFile);
		std::cout << "Embedding bichar file vocabulary size: " << m_driver._modelparams.ext_embeded_bichars.size() << std::endl;
	}
	else {
		std::cout << "bichar embedding file not specified." << std::endl;
		exit(0);
	}
	m_driver._modelparams.embeded_bichars.initial(bichar_stat, m_options.bicharCutOff);
	std::cout << "fine tuned bichar vocabulary size: " << m_driver._modelparams.embeded_bichars.size() << std::endl;

	//word
	if (m_options.wordEmbFile != "") {
		m_driver._modelparams.ext_embeded_words.initial(m_options.wordEmbFile);
		std::cout << "Embedding word file vocabulary size: " << m_driver._modelparams.ext_embeded_words.size() << std::endl;
	}
	else {
		std::cout << "word embedding file not specified." << std::endl;
		exit(0);
	}

	//pos tags
	unordered_map<string, int> postag_stat;
	for (int idx = 0; idx < m_driver._hyperparams.postags.size(); idx++) {
		postag_stat[m_driver._hyperparams.postags.getTagName(idx)]++;
	}
	postag_stat[nullkey]++;
	m_driver._modelparams.embeded_tags.initial(postag_stat);


    vector<CStateItem> state(m_driver._hyperparams.maxlength + 1);
    vector<string> seg_results, tag_results;
	
    CAction answer;
    Metric eval_seg, eval_tag;
    int actionNum;
	eval_seg.reset(); eval_tag.reset();
    for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
        const Instance &instance = vecInsts[numInstance];
        actionNum = 0;
        state[actionNum].clear();
        state[actionNum].setInput(&instance.chars);
        while (!state[actionNum].IsTerminated()) {
            state[actionNum].getGoldAction(instance, answer, &m_driver._hyperparams);
            state[actionNum].move(&(state[actionNum + 1]), answer);
            actionNum++;
        }

        state[actionNum].getResults(seg_results, tag_results, &m_driver._hyperparams);

        instance.evaluate(seg_results, tag_results, eval_seg, eval_tag);

        if (!eval_seg.bIdentical() || !eval_tag.bIdentical()) {
            cout << "error state conversion!" << std::endl;
            exit(0);
        }

        if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
            if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
        }
        if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
            break;
    }

	unordered_map<string, int> action_stat;
	for (int idx = 0; idx < m_driver._hyperparams.postags.size(); idx++) {
		answer.set(CAction::SEP, idx);
		action_stat[answer.str(&m_driver._hyperparams)]++;
	}
	answer.set(CAction::APP, -1);
	action_stat[answer.str(&m_driver._hyperparams)]++;
	answer.set(CAction::FIN, -1);
	action_stat[answer.str(&m_driver._hyperparams)]++;

	m_driver._modelparams.embeded_actions.initial(action_stat);
    cout << numInstance << " " << endl;

    return 0;
}

void JSTagger::getGoldActions(const vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions) {
    vecActions.clear();

	Metric eval_seg, eval_tag;
    vector<CStateItem> state(m_driver._hyperparams.maxlength + 1);
	vector<string> seg_results, tag_results;
    CAction answer;
	eval_seg.reset(); eval_tag.reset();
    int numInstance, actionNum;
    vecActions.resize(vecInsts.size());
    for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
        const Instance &instance = vecInsts[numInstance];

        actionNum = 0;
        state[actionNum].clear();
        state[actionNum].setInput(&instance.chars);
        while (!state[actionNum].IsTerminated()) {
			state[actionNum].getGoldAction(instance, answer, &m_driver._hyperparams);
            vecActions[numInstance].push_back(answer);
            state[actionNum].move(&state[actionNum + 1], answer);
            actionNum++;
        }


        state[actionNum].getResults(seg_results, tag_results, &m_driver._hyperparams);

		instance.evaluate(seg_results, tag_results, eval_seg, eval_tag);

		if (!eval_seg.bIdentical() || !eval_tag.bIdentical()) {
			cout << "error state conversion!" << std::endl;
			exit(0);
		}

        if ((numInstance + 1) % m_options.verboseIter == 0) {
            cout << numInstance + 1 << " ";
            if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
                cout << std::endl;
            cout.flush();
        }
        if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
            break;
    }
}

void JSTagger::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile) {
    if (optionFile != "")
        m_options.load(optionFile);

    vector<Instance> trainInsts, devInsts, testInsts;
    m_pipe.readInstances(trainFile, trainInsts, m_driver._hyperparams.maxlength, m_options.maxInstance);
    if (devFile != "")
        m_pipe.readInstances(devFile, devInsts, m_driver._hyperparams.maxlength, m_options.maxInstance);
    if (testFile != "")
        m_pipe.readInstances(testFile, testInsts, m_driver._hyperparams.maxlength, m_options.maxInstance);

    vector<vector<Instance> > otherInsts(m_options.testFiles.size());
    for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
        m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_driver._hyperparams.maxlength, m_options.maxInstance);
    }

    createAlphabet(trainInsts);

    bool initial_successed = false;
    if (m_options.charEmbFile != "") {
        initial_successed = m_driver._modelparams.ext_char_table.initial(&m_driver._modelparams.ext_embeded_chars, m_options.charEmbFile, false, false);
        if (initial_successed) {
            m_options.charEmbSize = m_driver._modelparams.ext_char_table.nDim;
        }
    }
	else {
		std::cout << "char embedding file not specified." << std::endl;
		exit(0);
	}
	
    initial_successed = false;
    if (m_options.bicharEmbFile != "") {
        initial_successed = m_driver._modelparams.ext_bichar_table.initial(&m_driver._modelparams.ext_embeded_bichars, m_options.bicharEmbFile, false, false);
        if (initial_successed) {
            m_options.bicharEmbSize = m_driver._modelparams.ext_bichar_table.nDim;
        }
    }
	else {
		std::cout << "bichar embedding file not specified." << std::endl;
		exit(0);
	}

	initial_successed = false;
	if (m_options.wordEmbFile != "") {
		initial_successed = m_driver._modelparams.ext_word_table.initial(&m_driver._modelparams.ext_embeded_words, m_options.wordEmbFile, false, false);
		if (initial_successed) {
			m_options.wordEmbSize = m_driver._modelparams.ext_word_table.nDim;
		}
	}
	else {
		std::cout << "bichar embedding file not specified." << std::endl;
		exit(0);
	}


	m_driver._modelparams.char_table.initial(&m_driver._modelparams.embeded_chars, m_options.charEmbSize, true);
	m_driver._modelparams.bichar_table.initial(&m_driver._modelparams.embeded_bichars, m_options.bicharEmbSize, true);
	m_driver._modelparams.tag_table.initial(&m_driver._modelparams.embeded_tags, m_options.tagEmbSize, true);


    m_driver._hyperparams.setRequared(m_options);
    m_driver.initial();

    vector<vector<CAction> > trainInstGoldactions;
    getGoldActions(trainInsts, trainInstGoldactions);
    double bestFmeasure = -1.0;

    int inputSize = trainInsts.size();

    std::vector<int> indexes;
    for (int i = 0; i < inputSize; ++i)
        indexes.push_back(i);

    Metric eval, metric_seg_dev, metric_tag_dev, metric_seg_test, metric_tag_test;

    int maxIter = m_options.maxIter * (inputSize / m_options.batchSize + 1);
    int oneIterMaxRound = (inputSize + m_options.batchSize - 1) / m_options.batchSize;
    cout << "maxIter = " << maxIter << std::endl;
    int devNum = devInsts.size(), testNum = testInsts.size();

    vector<vector<string> > seg_results, tag_results;
    bool bCurIterBetter;
    vector<vector<string> > subInstances;
    vector<vector<CAction> > subInstGoldActions;

	m_options.showOptions();
	m_driver._hyperparams.print();

    for (int iter = 0; iter < maxIter; ++iter) {
        cout << "##### Iteration " << iter << std::endl;
        srand(iter);
        bool bEvaluate = false;

        if (m_options.batchSize == 1) {
            auto t_start_train = std::chrono::high_resolution_clock::now();
            eval.reset();
            bEvaluate = true;
            random_shuffle(indexes.begin(), indexes.end());
            cout << "random: " << indexes[0] << ", " << indexes[indexes.size() - 1] << std::endl;
            for (int idy = 0; idy < inputSize; idy++) {
                subInstances.clear();
                subInstGoldActions.clear();
                subInstances.push_back(trainInsts[indexes[idy]].chars);
                subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);

                double cost = m_driver.train(subInstances, subInstGoldActions);

                eval.overall_label_count += m_driver._eval.overall_label_count;
                eval.correct_label_count += m_driver._eval.correct_label_count;

                if ((idy + 1) % (m_options.verboseIter) == 0) {
                    auto t_end_train = std::chrono::high_resolution_clock::now();
                    cout << "current: " << idy + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy()
                              << ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
                }

                m_driver.updateModel();
            }
            {
                auto t_end_train = std::chrono::high_resolution_clock::now();
                cout << "current: " << iter + 1 << ", Correct(%) = " << eval.getAccuracy()
                          << ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
            }
        } else {
            eval.reset();
            auto t_start_train = std::chrono::high_resolution_clock::now();
            bEvaluate = true;
            for (int idk = 0; idk < (inputSize + m_options.batchSize - 1)/m_options.batchSize; idk++) {
                random_shuffle(indexes.begin(), indexes.end());
                subInstances.clear();
                subInstGoldActions.clear();
                for (int idy = 0; idy < m_options.batchSize; idy++) {
                    subInstances.push_back(trainInsts[indexes[idy]].chars);
                    subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);
                }
                double cost = m_driver.train(subInstances, subInstGoldActions);

                eval.overall_label_count += m_driver._eval.overall_label_count;
                eval.correct_label_count += m_driver._eval.correct_label_count;

                if ((idk + 1) % (m_options.verboseIter) == 0) {
                    auto t_end_train = std::chrono::high_resolution_clock::now();
                    cout << "current: " << idk + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy()
                              << ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
                }

                m_driver.updateModel();
            }

            {
                auto t_end_train = std::chrono::high_resolution_clock::now();
                cout << "current: " << iter + 1 << ", Correct(%) = " << eval.getAccuracy()
                          << ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
            }
        }

        if (bEvaluate && devNum > 0) {
            auto t_start_dev = std::chrono::high_resolution_clock::now();
            cout << "Dev start." << std::endl;
            bCurIterBetter = false;
			if (!m_options.outBest.empty()) {
				seg_results.clear();
				tag_results.clear();
			}
            metric_seg_dev.reset(); metric_tag_dev.reset();
            predict(devInsts, seg_results, tag_results);
            for (int idx = 0; idx < devInsts.size(); idx++) {
                devInsts[idx].evaluate(seg_results[idx], tag_results[idx], metric_seg_dev, metric_tag_dev);
            }
            auto t_end_dev = std::chrono::high_resolution_clock::now();
            cout << "Dev finished. Total time taken is: " << std::chrono::duration<double>(t_end_dev - t_start_dev).count() << std::endl;
            cout << "dev:" << std::endl;
			metric_seg_dev.print();
			metric_tag_dev.print();

            if (!m_options.outBest.empty() && metric_tag_dev.getAccuracy() > bestFmeasure) {
                m_pipe.outputAllInstances(devFile + m_options.outBest, seg_results, tag_results);
                bCurIterBetter = true;
            }

            if (testNum > 0) {
                auto t_start_test = std::chrono::high_resolution_clock::now();
                cout << "Test start." << std::endl;
				if (!m_options.outBest.empty()) {
					seg_results.clear();
					tag_results.clear();
				}
                metric_seg_test.reset(); metric_tag_test.reset();
                predict(testInsts, seg_results, tag_results);
                for (int idx = 0; idx < testInsts.size(); idx++) {
					testInsts[idx].evaluate(seg_results[idx], tag_results[idx], metric_seg_test, metric_tag_test);
                }
                auto t_end_test = std::chrono::high_resolution_clock::now();
                cout << "Test finished. Total time taken is: " << std::chrono::duration<double>(t_end_test - t_start_test).count() << std::endl;
                cout << "test:" << std::endl;
                metric_seg_test.print();
				metric_tag_test.print();

                if (!m_options.outBest.empty() && bCurIterBetter) {
                    m_pipe.outputAllInstances(testFile + m_options.outBest, seg_results, tag_results);
                }
            }

            for (int idx = 0; idx < otherInsts.size(); idx++) {
                auto t_start_other = std::chrono::high_resolution_clock::now();
                cout << "processing " << m_options.testFiles[idx] << std::endl;
				if (!m_options.outBest.empty()) {
					seg_results.clear();
					tag_results.clear();
				}
				metric_seg_test.reset(); metric_tag_test.reset();
                predict(otherInsts[idx], seg_results, tag_results);
                for (int idy = 0; idy < otherInsts[idx].size(); idy++) {
                    otherInsts[idx][idy].evaluate(seg_results[idy], tag_results[idy], metric_seg_test, metric_tag_test);;
                }
                auto t_end_other = std::chrono::high_resolution_clock::now();
                cout << "Test finished. Total time taken is: " << std::chrono::duration<double>(t_end_other - t_start_other).count() << std::endl;
                cout << "test:" << std::endl;
				metric_seg_test.print();
				metric_tag_test.print();

                if (!m_options.outBest.empty() && bCurIterBetter) {
                    m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, seg_results, tag_results);
                }
            }


            if (m_options.saveIntermediate && metric_tag_dev.getAccuracy() > bestFmeasure) {
                cout << "Exceeds best previous DIS of " << bestFmeasure << ". Saving model file.." << std::endl;
                bestFmeasure = metric_tag_dev.getAccuracy();
                writeModelFile(modelFile);
            }
        }
    }
}

void JSTagger::predict(const vector<Instance>& inputs, vector<vector<string> >& seg_results, vector<vector<string> >& tag_results) {
    int sentNum = inputs.size();
    if (sentNum <= 0) return;
	seg_results.resize(sentNum);
	tag_results.resize(sentNum);
    vector<vector<string> > batch_sentences;
    vector<vector<string> > batch_seg_outputs, batch_tag_outputs;
    int processed_count = 0;
    for (int idx = 0; idx < sentNum; idx++) {
        batch_sentences.push_back(inputs[idx].chars);
        if (batch_sentences.size() == m_options.batchSize || idx == sentNum - 1) {
            m_driver.decode(batch_sentences, batch_seg_outputs, batch_tag_outputs);
            batch_sentences.clear();
            for (int idy = 0; idy < batch_seg_outputs.size(); idy++) {
                for (int idz = 0; idz < batch_seg_outputs[idy].size(); idz++) {
					seg_results[processed_count].push_back(batch_seg_outputs[idy][idz]);
					tag_results[processed_count].push_back(batch_tag_outputs[idy][idz]);
                }
                processed_count++;
            }
        }
    }

    if (processed_count != sentNum) {
        cout << "decoded number not match" << std::endl;
    }

}

void JSTagger::test(const string& testFile, const string& outputFile, const string& modelFile) {
    loadModelFile(modelFile);
    vector<Instance> testInsts;
    m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

    vector<vector<string> > seg_results, tag_results;
    Metric metric_seg_test, metric_tag_test;
	metric_seg_test.reset(); metric_tag_test.reset();
    predict(testInsts, seg_results, tag_results);
    for (int idx = 0; idx < testInsts.size(); idx++) {
        testInsts[idx].evaluate(seg_results[idx], tag_results[idx], metric_seg_test, metric_tag_test);
    }
    cout << "test:" << std::endl;
    metric_seg_test.print();
	metric_tag_test.print();

    std::ofstream os(outputFile.c_str());

    for (int idx = 0; idx < testInsts.size(); idx++) {
        for (int idy = 0; idy < seg_results[idx].size(); idy++) {
            os << seg_results[idx][idy] << "_" << tag_results[idx][idy] << " ";
        }
        os << std::endl;

    }
    os.close();
}


void JSTagger::loadModelFile(const string& inputModelFile) {

}

void JSTagger::writeModelFile(const string& outputModelFile) {

}

int main(int argc, char* argv[]) {
    std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
    std::string wordEmbFile = "", optionFile = "";
    std::string outputFile = "";
    bool bTrain = false;
    dsr::Argument_helper ah;
//    int threads = 2;


    ah.new_flag("l", "learn", "train or test", bTrain);
    ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
    ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
    ah.new_named_string("test", "testCorpus", "named_string",
                        "testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
    ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
    ah.new_named_string("word", "wordEmbFile", "named_string", "pretrained word embedding file to train a model, optional when training", wordEmbFile);
    ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
    ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
//    ah.new_named_int("th", "thread", "named_int", "number of threads for openmp", threads);

    ah.process(argc, argv);

//  omp_set_num_threads(threads);

    JSTagger tagger;
    if (bTrain) {
		tagger.train(trainFile, devFile, testFile, modelFile, optionFile);
    } else {
		tagger.test(testFile, outputFile, modelFile);
    }


}
