# Placeholder for numerology and astrology logic extracted from Streamlit app

def calculate_life_path(number: int) -> int:
    while number > 9:
        number = sum(int(digit) for digit in str(number))
    return number
