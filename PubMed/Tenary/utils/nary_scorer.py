

def score(key, prediction, cross_list=list()):
    test = False
    if len(cross_list) > 0:
        test = True

    assert len(key) == len(prediction)
    if test:
        assert len(cross_list) == len(prediction)
        dev_right = 0
        dev_single_right = 0
        dev_single_total = 0
        dev_total = len(prediction)
        for idx, pred in enumerate(prediction):
            if not cross_list[idx]:
                dev_single_total += 1
                if key[idx] == pred:
                    dev_right += 1
                    dev_single_right += 1
            else:
                if key[idx] == pred:
                    dev_right += 1
        dev_score = 1.0 * dev_right / dev_total
        dev_single = 1.0 * dev_single_right / dev_single_total
    else:
        dev_right = 0
        dev_total = len(prediction)
        for idx, pred in enumerate(prediction):
            if key[idx] == pred:
                dev_right += 1

        dev_score = 1.0 * dev_right / dev_total
        dev_single = 0

    return dev_score, dev_single
