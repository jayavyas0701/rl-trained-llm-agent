import pytest

from grpo_agent.data import extract_answer, gsm8k_answer
from grpo_agent.rewards import reward
from grpo_agent.tools import calculator


def test_calculator_arithmetic():
    assert calculator("15*12") == "180"
    assert calculator("(48 + 2) / 4") == "12.5"
    assert calculator("2**10") == "1024"
    assert calculator("1,200 - 200") == "1000"
    assert calculator("7 / 3") == "2.333333"


def test_calculator_is_safe():
    for bad in ["__import__('os').system('ls')", "open('x')", "a+1", "2**999999", "1/0", ""]:
        assert calculator(bad).startswith("error")


def test_answer_extraction():
    assert extract_answer("so 3 + 4 = 7. Answer: 7") == "7"
    assert extract_answer("Answer: $1,250.50") == "1250.50"
    assert extract_answer("Answer: 3 ... wait. Answer: 5") == "5"
    assert extract_answer("no final line") is None
    assert gsm8k_answer("blah <<2*3=6>>6\n#### 1,234") == "1234"


def test_reward_levels():
    assert reward("Answer: 12", "12", 1, 0)[0] == pytest.approx(1.2)       # correct + format + tool
    assert reward("Answer: 12", "12", 0, 0)[0] == pytest.approx(1.1)       # correct, no tool
    assert reward("Answer: 11", "12", 3, 0)[0] == pytest.approx(0.1)       # tool calls do not pay without a correct answer
    assert reward("I think 12", "12", 1, 0)[0] == pytest.approx(0.0)       # no answer line
