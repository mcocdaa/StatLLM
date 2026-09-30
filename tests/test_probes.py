import pytest
from statllm.probes import PROBES, get_probe

def test_q1_int_parsing():
    p = get_probe("q1_int")
    assert p is not None
    
    # Clean output
    res = p.parse("1. 42")
    assert res["is_valid"] is True
    assert res["parsed_value"] == "42"
    assert res["strictly_complied"] is True

    # Extra prefix / Chinese punctuation
    res2 = p.parse("1、73")
    assert res2["is_valid"] is True
    assert res2["parsed_value"] == "73"

    # With rambling explanation
    res3 = p.parse("好的，根据您的要求，我生成了一个随机数：1. 88。希望对您有帮助！")
    assert res3["is_valid"] is True
    assert res3["parsed_value"] == "88"
    assert res3["strictly_complied"] is False

    # Out of bounds
    res4 = p.parse("1. 999")
    assert res4["is_valid"] is False
    assert res4["parsed_value"] == "OUT_OF_BOUNDS"

    # Completely invalid
    res5 = p.parse("我无法为您生成随机数")
    assert res5["is_valid"] is False
    assert res5["parsed_value"] == "INVALID"


def test_q2_color_parsing():
    p = get_probe("q2_color")
    assert p is not None

    res1 = p.parse("红色")
    assert res1["parsed_value"] == "红色"
    assert res1["strictly_complied"] is True

    res2 = p.parse("我随机选择了蓝色。")
    assert res2["parsed_value"] == "蓝色"
    assert res2["strictly_complied"] is False


def test_q3_rps_parsing():
    p = get_probe("q3_rps")
    assert p is not None

    res1 = p.parse("石头")
    assert res1["parsed_value"] == "石头"
    assert res1["strictly_complied"] is True

    res2 = p.parse("剪刀")
    assert res2["parsed_value"] == "剪刀"


def test_q4_letter_parsing():
    p = get_probe("q4_letter")
    assert p is not None

    res1 = p.parse("M")
    assert res1["parsed_value"] == "M"
    assert res1["strictly_complied"] is True

    res2 = p.parse("The random letter is X.")
    assert res2["parsed_value"] == "X"
    assert res2["strictly_complied"] is False


def test_q5_sequence_parsing():
    p = get_probe("q5_sequence")
    assert p is not None

    res1 = p.parse("3, 7, 2")
    assert res1["parsed_value"] == "3,7,2"
    assert res1["strictly_complied"] is True

    res2 = p.parse("3，7，2")  # Full width comma
    assert res2["parsed_value"] == "3,7,2"
