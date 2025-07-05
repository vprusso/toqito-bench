import pytest
import matlab.engine

@pytest.fixture(scope="session")
def matlab_eng():
    eng = matlab.engine.start_matlab()
    yield eng
    eng.quit()

def test_matlab_add(matlab_eng):
    result = matlab_eng.plus(2, 3)
    assert result == 5
