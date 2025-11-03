import requests
import json
import time
import os
import pytest

# --- Configuration ---
API_URL = "http://localhost:8000/api/rag"

try:
    from app.config import DATA_TEST_FILE, DATA_TEST_FILE_DIR, DATA_DIR
    TEST_FILE = f"{DATA_DIR}/{DATA_TEST_FILE_DIR}/{DATA_TEST_FILE}"
    if not os.path.exists(TEST_FILE):
        QUERIES_FILE = f"../data/{DATA_TEST_FILE_DIR}/test.json"
except ImportError:
    print("Warning: Could not import from app.config. Falling back to relative path.")
    QUERIES_FILE = "../data/test_docs/test.json"

# --- Test Data Loading ---
def load_test_cases():
    """Loads test cases from the JSON file."""
    try:
        with open(TEST_FILE, 'r') as f:
            test_cases = json.load(f)
            if not isinstance(test_cases, list):
                pytest.fail(f"Test data in {TEST_FILE} is not a JSON list.")
            return test_cases
    except FileNotFoundError:
        pytest.fail(f"ERROR: Queries file not found at '{TEST_FILE}'")
    except json.JSONDecodeError:
        pytest.fail(f"ERROR: Could not decode JSON from '{TEST_FILE}'")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred loading test data: {e}")

ALL_TEST_CASES = load_test_cases()


# --- Pytest Test Function ---
@pytest.mark.parametrize("test_case", ALL_TEST_CASES)
def test_rag_api_query(test_case):
    """
    Sends a single query to the RAG API and asserts the response.
    """
    question = test_case.get("question")
    
    assert question is not None, f"Test case is missing 'question' field: {test_case}"
    
    payload = {"query": question}
    
    print(f"\n--- Testing query: {question} ---")

    try:
        tick = time.time()
        response = requests.post(API_URL, json=payload, timeout=60)
        tock = time.time()
        
        print(f"  Status Code: {response.status_code}")
        print(f"  Elapsed time: {tock - tick:.2f}s")
        
        assert response.status_code == 200, f"API returned non-200 status {response.status_code}"

        response_data = response.json()
        print(f"  Response: {response_data}")

        assert "answer" in response_data, "Response JSON is missing the 'answer' key"
        assert isinstance(response_data["answer"], str), "The 'answer' should be a string"
        assert len(response_data["answer"]) > 0, "The 'answer' string is empty"

    except requests.exceptions.RequestException as e:
        pytest.fail(f"FAILED: Could not connect to the server. Error: {e}")
    except json.JSONDecodeError:
        pytest.fail(f"FAILED: Could not decode JSON response from server. Response text: {response.text}")