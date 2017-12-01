/*
 * CAction.h
 *
 *  Created on: Jan 25, 2016
 *      Author: mszhang
 */

#ifndef BASIC_CAction_H_
#define BASIC_CAction_H_



/*===============================================================
 *
 * scored actions
 *
 *==============================================================*/
// for segmentation, there are only threee valid operations
class CAction {
  public:
    enum CODE { SEP = 0, APP = 1, FIN = 2, NO_ACTION = 3};
    unsigned long _code;
    int _tag;
    dtype _delta;

  public:
    CAction() : _code(NO_ACTION), _tag(-1), _delta(0) {
    }

    CAction(int code, int tag) : _code(code), _tag(tag), _delta(0) {
    }

    CAction(int code, int tag, dtype delta) : _code(code), _tag(tag), _delta(delta) {
    }

    CAction(const CAction &ac) : _code(ac._code), _tag(ac._tag), _delta(ac._delta) {
    }

  public:
    inline void clear() {
        _code = NO_ACTION;
        _tag = -1;
        _delta = 0;
    }

    inline void set(int code, int tag) {
        _code = code;
        _tag = tag;
        _delta = 0;
    }

    inline void set(int code, int tag, dtype delta) {
        _code = code;
        _tag = tag;
        _delta = delta;
    }

    inline void set(const CAction &ac) {
        _code = ac._code;
        _tag = ac._tag;
        _delta = ac._delta;
    }

    inline bool isNone() const {
        return _code==NO_ACTION && _tag == -1;
    }
    inline bool isSeparate() const {
        return _code==SEP && _tag >= 0;
    }
    inline bool isAppend() const {
        return _code==APP && _tag == -1;
    }
    inline bool isFinish() const {
        return _code==FIN && _tag == -1;
    }

  public:
    //ignore the delta
    inline std::string str(HyperParams *opt) const {
        if (isNone()) {
            return nullkey;
        }
        if (isSeparate()) {
            return "SEP_" + opt->postags.getTagName(_tag);
        }
        if (isAppend()) {
            return "APP";
        }
        if (isFinish()) {
            return "FIN";
        }
        return nullkey;
    }

  public:
    //ignore the delta
    bool operator == (const CAction &a1) const {
        return _code == a1._code && _tag == a1._tag;
    }
    bool operator != (const CAction &a1) const {
        return _code != a1._code || _tag != a1._tag;
    }

};


#endif /* BASIC_CAction_H_ */
