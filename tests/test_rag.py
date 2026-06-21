import pytest
import os
import sys

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import clean_text

def test_clean_text():
    # Test lowercase
    assert clean_text("HELLO") == "hello"
    
    # Test special characters removal
    assert clean_text("hello #world!") == "hello world!"
    
    # Test redaction label removal (XX/XX/XXXX)
    assert clean_text("date xx/xx/xxxx details") == "date details"
    
    # Test boilerplate removal
    assert clean_text("I am writing to file a complaint about my card") == "about my card"
    
    # Test spaces normalization
    assert clean_text("hello   world") == "hello world"
