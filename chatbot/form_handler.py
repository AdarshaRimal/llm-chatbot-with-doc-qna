import pandas as pd
from datetime import datetime
from chatbot.utils import parse_natural_date, is_valid_email, is_valid_phone

FORM_STEPS = ["name", "email", "phone", "date", "confirm"]

class AppointmentForm:
    FORM_STEPS = ["name", "email", "phone", "date", "confirm"]
    def __init__(self):
        self.current_step = None
        self.data = {}

    def agent_trigger(self, query: str) -> str:
        """Called by agent when appointment needed"""
        self.start()
        return "Let's start your appointment booking.\nWhat is your full name?"

    def start(self):
        self.current_step = 0
        self.data = {}

    @property
    def active(self):
        return self.current_step is not None and self.current_step < len(FORM_STEPS)

    def handle_input(self, user_input: str) -> tuple[str, bool]:
        if not self.active:
            return "", False

        step_name = FORM_STEPS[self.current_step]
        response = ""
        completed = False

        try:
            if step_name == "name":
                self.data["name"] = user_input.strip()
                response = "Please enter your email address:"
                self.current_step += 1

            elif step_name == "email":
                if not is_valid_email(user_input):
                    raise ValueError("Invalid email format")
                self.data["email"] = user_input.strip()
                response = "Please enter your phone number (with country code):"
                self.current_step += 1

            elif step_name == "phone":
                if not is_valid_phone(user_input):
                    raise ValueError("Invalid phone number")
                self.data["phone"] = user_input.strip()
                response = "When should we schedule? (e.g. 'next Monday' or YYYY-MM-DD):"
                self.current_step += 1

            elif step_name == "date":
                date = parse_natural_date(user_input)
                if not date:
                    raise ValueError("Couldn't understand date")
                # Check if date is in the future (date only)
                today = datetime.now().date()
                if date.date() < today:
                    raise ValueError("Date cannot be in the past")
                self.data["date"] = date.strftime("%Y-%m-%d")
                response = f"Confirm details:\n{self._format_details()}\nReply YES/NO"
                self.current_step += 1

            elif step_name == "confirm":
                if user_input.lower() in ["yes", "y"]:
                    self._save()
                    response = "✅ Appointment booked!"
                    completed = True
                else:
                    response = "❌ Booking cancelled"
                self.reset()

        except Exception as e:
            response = f"⚠️ Error: {str(e)}. Please try again:"

        return response, completed

    def _format_details(self):
        return "\n".join(
            f"{k.capitalize()}: {v}"
            for k, v in self.data.items()
        )

    def _save(self):
        record = self.data | {"timestamp": datetime.now().isoformat()}
        pd.DataFrame([record]).to_csv(
            "outputs/appointments.csv",
            mode="a",
            header=not pd.io.common.file_exists("outputs/appointments.csv"),
            index=False
        )

    def reset(self):
        self.current_step = None
        self.data = {}

    def should_start(self, user_input: str) -> bool:
        """Check if user wants to book appointment"""
        keywords = ["appointment", "schedule", "book a meeting", "call", "consultation"]
        return any(kw in user_input.lower() for kw in keywords)
