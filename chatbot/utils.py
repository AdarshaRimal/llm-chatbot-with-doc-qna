import re
import logging
from datetime import datetime,timedelta
import phonenumbers
from email_validator import validate_email, EmailNotValidError
from dateparser.search import search_dates
logger = logging.getLogger(__name__)

def is_valid_email(email: str) -> bool:
    try:
        validate_email(email)
        return True
    except EmailNotValidError as e:
        logger.warning(f"Invalid email: {email} - {str(e)}")
        return False

def is_valid_phone(number: str) -> bool:
    try:
        parsed = phonenumbers.parse(number, None)
        return phonenumbers.is_valid_number(parsed)
    except phonenumbers.NumberParseException as e:
        logger.warning(f"Invalid phone: {number} - {str(e)}")
        return False

def parse_natural_date(text: str) -> datetime | None:
    results = search_dates(
        text,
        settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": datetime.now(),
            "RETURN_AS_TIMEZONE_AWARE": False
        },
        languages=["en"]
    )

    if results:
        original_text, date = results[0]
        print(f"[DEBUG] Input: '{text}' → Parsed: '{original_text}' = {date}")

        today = datetime.now().date()


        if re.search(r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text.lower()):
            if date.date() <= today:
                print(f"[DEBUG] Detected past date for 'next weekday'. Bumping by 7 days.")
                date += timedelta(days=7)

        return date
    else:
        print(f"[DEBUG] No date parsed for input: '{text}'")
    return None