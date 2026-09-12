import math
import pytest
from scripts.audited_answer_checks import NumericReference,check_numeric,parse_real,single_choice

def test_exact_vs_verified_rounded_reference():
    assert check_numeric('-pi/3',NumericReference('-1.047','rounded',3)) is True
    assert check_numeric('-pi/3',NumericReference('-1.047','exact')) is False

def test_unverified_reference_or_required_method_stays_review():
    assert check_numeric('-1',NumericReference('-1')) is None
    assert check_numeric('-1.167',NumericReference('-1.167','exact',method_required=True)) is None

def test_declared_rounding_not_blanket_tolerance():
    assert check_numeric('10.067',NumericReference('10.0657784357283','rounded',3)) is False
    assert check_numeric('10.066',NumericReference('10.0657784357283','rounded',3)) is True
    assert check_numeric('30.159',NumericReference('30.159289474462','rounded',3)) is True
    assert check_numeric('.1855',NumericReference('.186','rounded',3)) is True

@pytest.mark.parametrize('text',['A','A. 77.07','(A)','Option A: 77.07'])
def test_single_choice_decorations(text): assert single_choice(text)=='A'

@pytest.mark.parametrize('text',['B,C','A or B','B and C','77.07','unknown'])
def test_ambiguous_choices_not_guessed(text): assert single_choice(text) is None

@pytest.mark.parametrize('s',["__import__('os').system('echo BAD')",'(1).__class__','10**(10**10)','sqrt(-1)','1/0','[1][0]'])
def test_unsafe_and_unbounded_numeric_rejected(s): assert parse_real(s) is None

def test_safe_numeric_forms():
    assert parse_real(r'\frac{1}{8}')==.125
    assert parse_real('2e-3')==.002
    assert math.isclose(parse_real('0.5*erfc(2/sqrt(10))'),.18554668476134878)
