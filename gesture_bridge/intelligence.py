"""Context-aware phrase generation for the offline prototype."""


class ContextInterpreter:
    CONTEXTS = ("General", "Home", "Hospital", "Classroom", "Public Office", "Transport")
    PHRASES = {
        "General": {
            "Hello": "Hello.", "Help": "I need help.", "Yes": "Yes.", "No": "No.",
            "Doctor": "Please call a doctor.", "Emergency": "This is an emergency.",
            "Caregiver": "Please contact my caregiver.",
        },
        "Home": {
            "Hello": "Hello, I am here.", "Help": "I need help at home.",
            "Yes": "Yes, that is okay.", "No": "No, please stop.",
            "Doctor": "Please call my doctor.",
            "Emergency": "Emergency at home. Contact my caregiver now.",
            "Caregiver": "Please come and check on me.",
        },
        "Hospital": {
            "Hello": "Hello, I need assistance.", "Help": "I need medical assistance.",
            "Yes": "Yes, that is correct.", "No": "No, please stop.",
            "Doctor": "Please call a doctor.",
            "Emergency": "Medical emergency. Please call a doctor now.",
            "Caregiver": "Please contact my caregiver.",
        },
        "Classroom": {
            "Hello": "Good morning.", "Help": "I need help with this lesson.",
            "Yes": "Yes, I understand.", "No": "No, I do not understand.",
            "Doctor": "I need to visit the medical room.",
            "Emergency": "Medical emergency in the classroom. Get help now.",
            "Caregiver": "Please contact my parent or caregiver.",
        },
        "Public Office": {
            "Hello": "Hello, I need assistance at this counter.",
            "Help": "Please help me with this process.",
            "Yes": "Yes, this information is correct.",
            "No": "No, this information is not correct.",
            "Doctor": "I need medical assistance.",
            "Emergency": "Emergency. Please contact security and medical help.",
            "Caregiver": "Please contact my caregiver or companion.",
        },
        "Transport": {
            "Hello": "Hello, I need travel assistance.",
            "Help": "Please help me with this route or vehicle.",
            "Yes": "Yes, this is my destination.", "No": "No, this is not my destination.",
            "Doctor": "I need medical assistance while travelling.",
            "Emergency": "Transport emergency. Stop safely and call for help.",
            "Caregiver": "Contact my caregiver and share my location.",
        },
    }

    def __init__(self):
        self.context_index = 0

    @property
    def context(self):
        return self.CONTEXTS[self.context_index]

    def cycle(self):
        self.context_index = (self.context_index + 1) % len(self.CONTEXTS)
        return self.context

    def interpret(self, gesture):
        return self.phrase_for(self.context, gesture)

    @classmethod
    def phrase_for(cls, context, gesture):
        return cls.PHRASES.get(context, {}).get(gesture, gesture)

    def compose(self, gestures, limit=3):
        """Compose recent commands using the vocabulary of the active setting."""
        unique = []
        for gesture in gestures:
            if not unique or unique[-1] != gesture:
                unique.append(gesture)
        if not unique:
            return "Waiting for a stable sign..."
        return " ".join(self.interpret(gesture) for gesture in unique[-limit:])


class SentenceEngine:
    """Small deterministic grammar layer suitable for offline edge use."""

    def __init__(self, limit=6):
        self.tokens = []
        self.limit = limit

    def clear(self):
        self.tokens.clear()

    def add(self, gesture):
        if not self.tokens or self.tokens[-1] != gesture:
            self.tokens.append(gesture)
            self.tokens = self.tokens[-self.limit:]
        return self.compose()

    def compose(self):
        lowered = [token.lower() for token in self.tokens]
        token_set = set(lowered)
        if {"pain", "chest"}.issubset(token_set):
            return "I am having chest pain. Please call a doctor."
        if {"call", "caregiver"}.issubset(token_set):
            return "Please call my caregiver."
        if "injury" in token_set:
            return "I am injured. Please help me."
        if "fever" in token_set:
            return "I have a fever. Please check my temperature."
        if {"need", "medicine"}.issubset(token_set) or "medicine" in token_set:
            return "I need my medicine."
        if "help" in token_set and "doctor" in token_set:
            return "I need help. Please call a doctor."
        if "emergency" in token_set:
            return "This is an emergency. Please alert a caregiver."
        if not self.tokens:
            return "Waiting for a stable sign..."
        replacements = {"Thank You": "thank you", "Yes": "yes", "No": "no"}
        words = [replacements.get(token, token.lower()) for token in self.tokens]
        return " ".join(words).capitalize() + "."
