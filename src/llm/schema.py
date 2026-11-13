"""Shared annotation vocabulary. Version 1 preserves the source stance labels."""

STANCE_LABELS = ("Positive", "Negative", "Neutral", "Mixed")
TOPICS = (
    "Anxiety", "Depression", "Suicide/Self-Harm", "Bullying/Cyberbullying",
    "Body Image/Eating Disorders", "Addiction/Overuse", "Sleep Disruption",
    "Policy/Lawsuits/Bans", "School/Education", "Therapy/Access to Care",
    "Stigma/Awareness", "Parental Controls/Safety Features",
)


def validate_prediction(value):
    if not isinstance(value, dict) or set(value) != {"label", "topics", "reason"}:
        raise ValueError("Expected exactly label, topics, and reason")
    if value["label"] not in STANCE_LABELS:
        raise ValueError("Unknown stance label")
    topics = value["topics"]
    if (not isinstance(topics, list) or any(not isinstance(t, str) for t in topics)
            or len(topics) > 3 or len(set(topics)) != len(topics)
            or any(t not in TOPICS for t in topics)):
        raise ValueError("Expected up to three unique topics from the vocabulary")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("A short reason is required")
    return value
