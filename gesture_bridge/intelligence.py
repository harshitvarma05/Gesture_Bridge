"""Context-aware phrase generation for the offline prototype."""


class ContextInterpreter:
    CONTEXTS = ("General", "Hospital", "Classroom", "Public Office")
    PHRASES = {
        "General": {"Help": "I need help.", "Emergency": "This is an emergency.", "Pain": "I am in pain.", "Injury": "I am injured.", "Fever": "I have a fever.", "Drink": "I need a drink.", "Cry": "I am upset.", "Call": "Please make a call."},
        "Hospital": {"Help": "I need medical assistance.", "Doctor": "Please call a doctor.", "Water": "May I have some water?", "Medicine": "I need my medicine.", "Pain": "I am in pain.", "Injury": "I have an injury and need assistance.", "Fever": "I have a fever. Please check my temperature.", "Drink": "May I have something to drink?", "Cry": "I am distressed and need assistance.", "Chest": "The pain is in my chest.", "Emergency": "Medical emergency. Please call a doctor."},
        "Classroom": {"Help": "I need help with this.", "Doctor": "I need to visit the medical room.", "Water": "May I drink some water?"},
        "Public Office": {"Help": "Please help me with this process.", "Doctor": "I need medical assistance.", "Emergency": "Emergency—please contact security."},
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
        return self.PHRASES.get(self.context, {}).get(gesture, gesture)


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
