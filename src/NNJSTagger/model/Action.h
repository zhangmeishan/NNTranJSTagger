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

  public:
    CAction() : _code(NO_ACTION), _tag(-1) {
    }

    CAction(int code, int tag) : _code(code), _tag(tag) {
    }

    CAction(const CAction &ac) : _code(ac._code), _tag(ac._tag) {
    }

  public:
    inline void clear() {
        _code = NO_ACTION;
        _tag = -1;
    }

    inline void set(int code, int tag) {
        _code = code;
        _tag = tag;
    }

    inline void set(const CAction &ac) {
        _code = ac._code;
        _tag = ac._tag;
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
    inline std::string str(HyperParams *opt) const {
        if (isNone()) {
            return nullkey;
        }
        if (isSeparate()) {
            return "SEP_" + opt->postags.from_id(_tag);
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
    bool operator == (const CAction &a1) const {
        return _code == a1._code && _tag == a1._tag;
    }
    bool operator != (const CAction &a1) const {
        return _code != a1._code || _tag != a1._tag;
    }

};


#endif /* BASIC_CAction_H_ */
