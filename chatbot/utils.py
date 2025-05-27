import re
import logging
from datetime import datetime
import dateparser
import phonenumbers
from email_validator import validate_email, EmailNotValidError

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

def parse_natural_date(text: str, anchor_date: datetime = None) -> str:
    settings = {
        'RELATIVE_BASE': anchor_date or datetime.now(),
        'PREFER_DATES_FROM': 'future'
    }
    parsed = dateparser.parse(text, settings=settings)
    return parsed.strftime("%Y-%m-%d") if parsed else None