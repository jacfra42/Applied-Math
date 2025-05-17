# test_specs.py
"""Python Essentials: Unit Testing.
Jacob Francis
Vol 2 Lab
4 Oct 2023
"""

import specs
import pytest


def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_factor():
    assert specs.smallest_factor(4) == 2, "failed on 1"
    assert specs.smallest_factor(101) == 101, "failed on prime numbers"
    assert specs.smallest_factor(37) == 37, "failed on odds"
    assert specs.smallest_factor(201) == 3, "failed even"

# Problem 2: write a unit test for specs.month_length().
def test_month():
    assert specs.month_length("April") == 30, "April failed"
    assert specs.month_length("July") == 31, "failed July"
    assert specs.month_length("February",True) == 29, "failed Feb leap year"
    assert specs.month_length("February",False) == 28, "failed Feb, not leap year"
    assert specs.month_length("Maytember") == None, "not a month failed"

# Problem 3: write a unit test for specs.operate().
def test_oper():
    assert specs.operate(2,3,'+') == 5, "failed add"
    assert specs.operate(3,2,'-') == 1, "failed subtract"
    assert specs.operate(2,2,'*') == 4, "failed multiply"
    assert specs.operate(4,2,'/') == 2, "failed simple divide"
    with   pytest.raises(ZeroDivisionError) as excinfo:
           specs.operate(4,0,'/')
    assert excinfo.value.args[0] == "division by zero is undefined"
    with   pytest.raises(TypeError) as excinfo:
           specs.operate(1,1,3)
    assert excinfo.value.args[0]  == "oper must be a string"
    with   pytest.raises(ValueError) as excinfo:
           specs.operate(1,1,'%')
    assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"

# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    frac_4_1 = specs.Fraction(4,1)
    return frac_1_3, frac_1_2, frac_n2_3, frac_4_1
# Check that the fractions are created correctly
def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_4_1 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.numer == 1
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7
    with pytest.raises(ZeroDivisionError) as excinfo:
         specs.Fraction(1,0)
    assert excinfo.value.args[0] == "denominator cannot be zero"
    with pytest.raises(TypeError) as excinfo:
         specs.Fraction('e',2)
    assert excinfo.value.args[0] == "numerator and denominator must be integers"
    with pytest.raises(TypeError) as excinfo:
         specs.Fraction(2,'d')
    assert excinfo.value.args[0] == "numerator and denominator must be integers"
# Check that fractions print correctly
def test_fraction_str(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_4_1 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    assert str(frac_4_1) == "4"
# Check that floats are returned
def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_4_1 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.
# Check equality
def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_4_1 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)
    assert frac_4_1 == 4
# Check add
def test_fraction_add(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_4_1 = set_up_fractions
    assert frac_1_2 + frac_1_3 == specs.Fraction(5, 6)
    assert frac_n2_3 + frac_4_1 == specs.Fraction(10, 3)
# Check sub
def test_fraction_sub(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_4_1 = set_up_fractions
    assert frac_1_2 - frac_1_3 == specs.Fraction(1, 6)
    assert frac_n2_3 - frac_4_1 == specs.Fraction(-14, 3)
# Check mult
def test_fraction_mul(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_4_1 = set_up_fractions
    assert frac_4_1 * frac_1_2 == specs.Fraction(2, 1)
    assert frac_n2_3 * frac_1_3 == specs.Fraction(-2, 9)
# Check div
def test_fraction_div(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_4_1 = set_up_fractions
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.Fraction(2,1)/specs.Fraction(0,2)
    assert excinfo.value.args[0] == "cannot divide by zero"
    assert frac_4_1 / frac_1_3 == specs.Fraction(12, 1)
        

# Problem 5: Write test cases for Set.
hand = ["1022", "1122", "0100", "2021", # A good hand with 6 sets
"0010", "2201", "2111", "0020",
"1102", "0200", "2110", "1020"]
hand_1 = ["122", "1122", "0100", "2021",  # Set with a card with 3 digits
"0010", "2201", "2111", "0020",
"1102", "0200", "2110", "1020"]
hand_2 = ["1022", "1122", "0100", "2021", # Set with 13 cards
"0010", "2201", "2111", "0020",
"1102", "0200", "2110", "1020", "2102"]   
hand_3 = ["1022", "1022", "0100", "2021", # Set where two cards are identical
"0010", "2201", "2111", "0020",
"1102", "0200", "2110", "1020"]
hand_4 = ["6022", "1022", "0100", "2021", # Set where a card contains something other than 0,1,2
"0010", "2201", "2111", "0020",
"1102", "0200", "2110", "2103"]

def test_count():
    # Test that first hand gets the right number of sets
    assert specs.count_sets(hand) == 6 
                                                                    # Sorry hands aren't in numerical order
                                                                    # Comments help understand what each test does 
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(hand_2)
    assert excinfo.value.args[0] == "Not exactly 12 cards"
    
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(hand_3)
    assert excinfo.value.args[0] == "Cards are not unique"
    
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(hand_1)
    assert excinfo.value.args[0] == "One or more cards doesn't have exactly 4 digits"
    
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(hand_4)
    assert excinfo.value.args[0] == "Cards must contain only '0', '1', or '2'"
def test_is():
    assert specs.is_set('1022','1022','1022') == True, "a,b,c do form a set"
    assert specs.is_set('1000','2110','2011') == False, "a,b,c are not a set"