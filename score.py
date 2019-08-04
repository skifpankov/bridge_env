from config import ScoreBias, ScoreScale

## This script is for scoring Bridge contracts in 3 possible ways:
##  v0 - original IMP, see https://tinyurl.com/ydcwepa9
##  v1 - original IMP with slam bonus updated to reflect real Bridge games
##  v2 - see https://en.wikipedia.org/wiki/Bridge_scoring (single table)
##  v3 - simplification - remove points for overtricks


def convert2IMP(diff):
    imp = 0
    if diff >= 20 and diff < 50:
        imp = 1
    elif diff >= 50 and diff < 90:
        imp = 2
    elif diff >= 90 and diff < 130:
        imp = 3
    elif diff >= 130 and diff < 170:
        imp = 4
    elif diff >= 170 and diff < 220:
        imp = 5
    elif diff >= 220 and diff < 270:
        imp = 6
    elif diff >= 270 and diff < 320:
        imp = 7
    elif diff >= 320 and diff < 370:
        imp = 8
    elif diff >= 370 and diff < 430:
        imp = 9
    elif diff >= 430 and diff < 500:
        imp = 10
    elif diff >= 500 and diff < 600:
        imp = 11
    elif diff >= 600 and diff < 750:
        imp = 12
    elif diff >= 750 and diff < 900:
        imp = 13
    elif diff >= 900 and diff < 1100:
        imp = 14
    elif diff >= 1100 and diff < 1300:
        imp = 15
    elif diff >= 1300 and diff < 1500:
        imp = 16
    elif diff >= 1500 and diff < 1750:
        imp = 17
    elif diff >= 1750 and diff < 2000:
        imp = 18
    elif diff >= 2000 and diff < 2250:
        imp = 19
    elif diff >= 2250 and diff < 2500:
        imp = 20
    elif diff >= 2500 and diff < 3000:
        imp = 21
    elif diff >= 3000 and diff < 3500:
        imp = 22
    elif diff >= 3500 and diff < 4000:
        imp = 23
    elif diff >= 4000:
        imp = 24
    return imp


def score_IMP(input_tuple, version="v0"):
    """ Calculates a variant of the IMP point score for a given bid and hand

    OPTIONS FOR *version* ARG:
      "v0" - original as implemented here https://tinyurl.com/ydcwepa9
      "v1" - slams updated only if possible and bid high enough
    """
    # bid_tricks range from 1 - 7
    # max_tricks range from 0 - 13
    bid_tricks, trump, max_tricks = input_tuple
    declarer_score = 0
    defender_score = 0

    # Major suit trump.
    # Suit2str = {0: "S", 1: "H", 2: "D", 3: "C"}
    # Did the declarer make at least 6 tricks?
    if max_tricks > 6:
        contract_tricks = min(max_tricks - 6, bid_tricks)
        declarer_score += (ScoreScale[trump] * contract_tricks + ScoreBias[trump])

    # Were there overtricks?
    if max_tricks > (bid_tricks + 6):
        over_tricks = max_tricks - bid_tricks - 6
        declarer_score += ScoreScale[trump] * over_tricks

    # Were there penalty points?
    if max_tricks < (bid_tricks + 6):
        under_tricks = bid_tricks + 6 - max_tricks
        defender_score += 50 * under_tricks

    # Was there a slam?
    if version == "v0":
        if max_tricks == 12:
            declarer_score += 500
        elif max_tricks == 13:
            declarer_score += 1000
    elif version == "v1":
        if max_tricks == 12 and bid_tricks >= 6:
            declarer_score += 500
        elif max_tricks == 13 and bid_tricks == 7:
            declarer_score += 1000

    # Converting to IMP
    diff = abs(declarer_score - defender_score)
    imp = convert2IMP(diff)

    if declarer_score > defender_score:
        return imp
    else:
        return (-1) * imp


def precompute_scores_IMP(version="v0"):
    """ Computes the score_bridge function for all possible inputs

    Args:
        version: 'str', either "v0" or "v1" - see line 4 and 5 of this script

    Returns:
        scorer: 'dict', a dictionary; keys = possible inputs, values = scores
    """

    assert version in {"v0", "v1"}, "invalid version for this function"

    input_space = [(bid_tricks, trump, max_tricks) for bid_tricks in range(1, 8)
                   for trump in range(5)
                   for max_tricks in range(14)]

    score_space = [score_IMP(t, version) for t in input_space]
    scorer = dict(zip(input_space, score_space))
    del input_space, score_space
    return scorer


def score_v2(bid_tricks: int,
             trump: int,
             actual_tricks: int,
             vul: bool = False,
             double: int = 0,
             norm: float = 1.0,
             keep_overtricks: bool = True,
             game: bool = False,
             score_offset: float = 0.0) -> float:
    """
    Scores a single Bridge table using the rules as described in the
    wikipedia article: https://en.wikipedia.org/wiki/Bridge_scoring

    :param bid_tricks: 'int', in [1,7]
    :param trump: 'int', in [0,4] - where 0,1,2,3,4 <=> C,D,H,S,NT
    :param actual_tricks: 'int', in [0,13] (tricks achieved/tricks possible)
    :param vul: 'bool', if the declarer is vulnerable
    :param double: 'int', where 0,1,2 <=> (NOT, DOUBLE, REDOUBLE) respectively
    :param norm: 'float', normalise the output score by dividing by this
    :param keep_overtricks:
    :param game:
    :param score_offset: float; an offset that should be added to the score before normalisation
    (one potential use of this is to set the minimum score to be 0)

    :return: 'float', corresponding Bridge points
    """

    assert bid_tricks in set(range(1, 8)), "invalid value for bid_tricks"
    assert trump in set(range(5)), "invalid value for trump"
    assert actual_tricks in set(range(14)), "invalid value for actual_tricks"
    assert vul in set(range(2)), "invalid value for vul"
    assert double in set(range(3)), "invalid value for double"

    score = 0

    delta = actual_tricks - (bid_tricks + 6)


    #### CONTRACT SUCCESSFUL ####
    if delta >= 0:

        # CONTRACT POINTS
        score += bid_tricks * ScoreScale[trump] + ScoreBias[trump]

        # Game Points
        if score >= 100:
            score += (300+200*vul)*game
        else:
            score += 50*game

        # DOUBLE/REDOUBLE
        score *= 2 ** double

        # INSULT BONUS
        score += 50 * double

        # SLAM BONUS
        if bid_tricks == 6 and delta >= 0:
            score += 500 + 250 * vul
        elif bid_tricks == 7 and delta == 0:
            score += 1000 + 500 * vul

        # OVERTRICKS
        if delta > 0:
            if double == 1:
                score += 100 * delta + vul * (100 * delta)
            elif double ==2:
                score += 200 * delta + vul * (200 * delta)
            else:
                score += delta * ScoreScale[trump] * keep_overtricks


    #### CONTRACT UNSUCCESSFUL ####
    else:
        under_tricks = delta * -1
        if vul:
            if double == 0:
                score = under_tricks * 100
            elif double == 1:
                if under_tricks == 1:
                    score = 200
                else:
                    score = 200 + 300 * (under_tricks - 1)
            else:
                if under_tricks == 1:
                    score = 400
                else:
                    score = 400 + 600 * (under_tricks - 1)

        else:
            if double == 0:
                score = under_tricks * 50
            elif double == 1:
                if under_tricks == 1:
                    score = 100
                elif under_tricks == 2 or under_tricks == 3:
                    score = 100 + 200 * (under_tricks - 1)
                elif under_tricks > 3:
                    score = 100 + 200 + 200 + 300 * (under_tricks - 3)
            else:
                if under_tricks == 1:
                    score = 200
                elif under_tricks == 2 or under_tricks == 3:
                    score = 200 + 400 * (under_tricks - 1)
                elif under_tricks > 3:
                    score = 200 + 400 + 400 + 600 * (under_tricks - 3)

        score = score * -1

    out: float = (score + score_offset) / norm

    return out


def precompute_scores_v2(full_version: bool = False,
                         normalise: bool = True,
                         min_zero: bool = False,
                         keep_overtricks: bool = True,
                         game: bool = False):
    """
    Computes the score_bridge function for all possible inputs

    :param full_version: whether vulnerabilities and doubles should be used when calculating scores
    :param normalise: whether the score should be normalised
    :param min_zero: whether the lowest score should be 0 or not
    :param keep_overtricks:
    :param game:

    :return: a dictionary of all possible bridge scores
    """

    score_offset = 0

    if full_version:
        #print("Generating score dictionary, including vul and double")
        input_space = [(bid_tricks, trump, actual_tricks, vul, double)
                        for bid_tricks in range(1, 8)
                        for trump in range(5)
                        for actual_tricks in range(14)
                        for vul in range(2)
                        for double in range(3)]

        if min_zero:
            score_offset = 7600
            norm = score_offset + 2660 + (500 + 1320) * game
        else:
            norm = 7600 + 500 * game

    else:
        #print("Generating score dictionary, without vul and double")
        input_space = [(bid_tricks, trump, actual_tricks)
                        for bid_tricks in range(1,8)
                        for trump in range(5)
                        for actual_tricks in range(14)]

        if min_zero:
            score_offset = 650
            norm = score_offset + 1220 + 300 * game
        else:
            norm = 1220 + 300 * game

    if not normalise:
        norm = 1

    # applying score_v2 to each tuple in input_space to get the score
    score_space = [score_v2(*t,
                            norm=norm,
                            keep_overtricks=keep_overtricks,
                            game=game,
                            score_offset=score_offset)
                   for t in input_space]

    scorer = dict(zip(input_space, score_space))

    del input_space, score_space

    return scorer


print("Calculating score tables")
SCORE_TABLE_v0 = precompute_scores_IMP()
SCORE_TABLE_v1 = precompute_scores_IMP(version="v1")
SCORE_TABLE_v2_simple = precompute_scores_v2(full_version=False, normalise=True)
SCORE_TABLE_v2_full = precompute_scores_v2(full_version=True, normalise=True)
SCORE_TABLE_v3 = precompute_scores_v2(full_version=False, normalise=True, keep_overtricks=False)  # overtrick score = 0
SCORE_TABLE_v4 = precompute_scores_v2(full_version=False, normalise=True, game=True)
print("SCORE_TABLE CALCULATIONS COMPLETED")
