import numpy as np
import pytest

from nanomaternalpfn.birthwt import BIRTHWT_FEATURES, parse_birthwt_csv


def test_birthwt_parser_excludes_birth_weight_target_source():
    text = (
        "rownames,low,age,lwt,race,smoke,ptl,ht,ui,ftv,bwt\n"
        "85,0,19,182,2,0,0,0,1,0,2523\n"
        "86,1,33,155,3,0,0,0,0,3,2551\n"
    )

    X, y = parse_birthwt_csv(text)

    assert BIRTHWT_FEATURES == (
        "age",
        "lwt",
        "race",
        "smoke",
        "ptl",
        "ht",
        "ui",
        "ftv",
    )
    assert X.shape == (2, 8)
    np.testing.assert_array_equal(y, np.array([0, 1]))
    assert 2523 not in X
    assert 2551 not in X


def test_birthwt_parser_rejects_nonbinary_target():
    text = (
        "rownames,low,age,lwt,race,smoke,ptl,ht,ui,ftv,bwt\n"
        "1,2,19,120,1,0,0,0,0,0,2500\n"
    )

    with pytest.raises(ValueError, match="binary"):
        parse_birthwt_csv(text)
