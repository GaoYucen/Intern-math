from scripts.r2_eval import strict, numeric, parse_judge, select

def test_no_substring_acceptance():
    assert strict({'answer_type':'symbolic','answer':'x^2+1','problem':'compute'},'FINAL_ANSWER: x^2+10') is None

def test_correct_numeric():
    assert strict({'answer_type':'integer','answer':72,'problem':'compute'},'FINAL_ANSWER: 72') is True
    assert numeric(r'\frac{1}{8}')==0.125

def test_proof_needs_judge():
    assert strict({'answer_type':'proof','answer':'True','problem':'prove'},'FINAL_ANSWER: True') is None

def test_judge_failures_stay_unresolved():
    assert parse_judge('no usable JSON')['A']=='unresolved'
    assert parse_judge('{"A":"correct","B":"incorrect"}')['B']=='incorrect'

def test_selection_order_invariant():
    rows=[{'domain':'d','problem':str(i),'idx':i} for i in range(10)]
    assert select(rows,2)==select(list(reversed(rows)),2)
