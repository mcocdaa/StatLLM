import pytest
from statllm.probes import PROBES, get_probe

def test_arr_int5_parsing():
    p = get_probe("arr_int5")
    assert p is not None

    # Clean JSON array
    res = p.parse("[42, 17, 89, 5, 63]")
    assert res["is_valid"] is True
    assert res["parsed_tokens"] == ["42", "17", "89", "5", "63"]
    assert res["strictly_complied"] is True
    assert res["traits"]["has_duplicates"] is False
    assert res["traits"]["first_token"] == "42"

    # With markdown code blocks or text
    res2 = p.parse("```json\n[42, 17, 89, 5, 63]\n```")
    assert res2["is_valid"] is True
    assert res2["parsed_tokens"] == ["42", "17", "89", "5", "63"]
    assert res2["strictly_complied"] is False

    # Duplicates detection
    res3 = p.parse("[42, 42, 10, 20, 30]")
    assert res3["traits"]["has_duplicates"] is True

    # Sorting bias detection
    res4 = p.parse("[5, 12, 45, 78, 99]")
    assert res4["traits"]["is_sorted"] is True

    # Invalid length
    res5 = p.parse("[1, 2, 3]")
    assert res5["is_valid"] is False


def test_arr_color5_parsing():
    p = get_probe("arr_color5")
    assert p is not None

    res = p.parse('["黄", "青", "红", "紫", "橙"]')
    assert res["is_valid"] is True
    assert res["parsed_tokens"] == ["黄", "青", "红", "紫", "橙"]
    assert res["traits"]["has_duplicates"] is False

    # English input support
    res_en = p.parse('["yellow", "cyan", "red", "purple", "orange"]')
    assert res_en["is_valid"] is True
    assert res_en["parsed_tokens"] == ["黄", "青", "红", "紫", "橙"]


def test_arr_rps5_parsing():
    p = get_probe("arr_rps5")
    assert p is not None

    res = p.parse('["布", "剪刀", "石头", "布", "剪刀"]')
    assert res["is_valid"] is True
    assert res["parsed_tokens"] == ["布", "剪刀", "石头", "布", "剪刀"]
    assert res["traits"]["has_duplicates"] is True

    # English input support
    res_en = p.parse('["paper", "scissors", "rock", "paper", "scissors"]')
    assert res_en["is_valid"] is True
    assert res_en["parsed_tokens"] == ["布", "剪刀", "石头", "布", "剪刀"]


def test_arr_letter5_parsing():
    p = get_probe("arr_letter5")
    assert p is not None

    res = p.parse('["K", "W", "B", "R", "M"]')
    assert res["is_valid"] is True
    assert res["parsed_tokens"] == ["K", "W", "B", "R", "M"]


def test_arr_perm5_parsing():
    p = get_probe("arr_perm5")
    assert p is not None

    res = p.parse('[4, 1, 5, 3, 2]')
    assert res["is_valid"] is True
    assert res["parsed_tokens"] == ["4", "1", "5", "3", "2"]
    assert res["traits"]["canonical_perm"] == "4,1,5,3,2"
