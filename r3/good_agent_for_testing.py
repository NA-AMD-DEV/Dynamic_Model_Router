# good_agent_for_testing.py -- NOT part of the deliverable.
# Exists only to prove llm_judge.py can actually pass a correct agent, not
# just fail everything. Delete or ignore this file; it's a test fixture.

ANSWERS = {
    "fact-01": "The capital of Australia is Canberra, which sits near Lake Burley Griffin.",
    "fact-02": "Photosynthesis is the process by which plants convert sunlight into chemical energy, using water and carbon dioxide. This produces glucose and releases oxygen as a byproduct.",
    "fact-03": "Seasons are caused by Earth's axial tilt of about 23.5 degrees relative to its orbit around the sun. This tilt means different hemispheres receive more direct sunlight at different times of year.",
    "fact-04": "Albert Einstein published the theory of general relativity in 1915.",
    "math-01": "240 - 36 (15%) - 60 = 144 items remain.",
    "math-02": "Speed is 80 km/h, so 180 km takes 135 minutes.",
    "math-03": "Perimeter is 40 meters, plus 1 meter overlap = 41 meters of fencing.",
    "math-04": "500 * 1.08^3 is about 629.86, which rounds to 630 users.",
    "sent-01": "This is mixed sentiment: positive about battery life, negative about the screen scratching easily.",
    "sent-02": "negative, because of the long wait, cold food, and no apology from staff.",
    "sent-03": "positive, given the strong praise and reliability mentioned.",
    "sent-04": "neutral, the tone is lukewarm and unremarkable.",
    "sum-01": "The city council voted 6-3 to approve a light rail extension from downtown to the northern suburbs, costing about 480 million dollars and starting construction in 2027.",
    "sum-02": "A new battery chemistry charges to 80 percent in under ten minutes and retains over 90 percent capacity after 1000 cycles, with possible EV use within five years pending verification.",
    "sum-03": "Festival drew 200,000+ attendees over three rainy days, boosting local business revenue.",
    "sum-04": "A ten-year study of 5,000 people found 7,000+ daily steps linked to lower early death risk than under 4,000 steps. Benefits plateaued around 10,000 steps.",
    "ner-01": "Maria Sanchez (PERSON), Fireworks AI (ORG), Berlin (LOCATION), last March (DATE)",
    "ner-02": "July 4th (DATE), President Okafor (PERSON), United Nations (ORG), Geneva (LOCATION)",
    "ner-03": "Amazon (ORG), last Friday (DATE), Seattle (LOCATION), Dana Wu (PERSON)",
    "ner-04": "MIT (ORG), Cambridge (LOCATION), Dr. Elena Petrov (PERSON), March 3rd (DATE)",
    "debug-01": "```python\ndef get_max(nums):\n    return max(nums)\n```",
    "debug-02": "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
    "debug-03": "```python\ndef average(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total / len(nums)\n```",
    "debug-04": "```python\ndef reverse_string(s):\n    return s[::-1]\n```",
    "logic-01": "Sam owns the cat, since Jo owns the dog and Sam doesn't own the bird.",
    "logic-02": "Ana sits at desk 1.",
    "logic-03": "No, Maya cannot be an engineer, since all engineers know Python and Maya doesn't.",
    "logic-04": "12 red balls, since the 2:1 ratio over 18 total gives 12 red and 6 blue.",
    "gen-01": "```python\ndef second_largest(nums):\n    uniq = sorted(set(nums), reverse=True)\n    return uniq[1]\n```",
    "gen-02": "```python\nimport re\ndef is_palindrome(s):\n    cleaned = re.sub(r'[^a-z0-9]', '', s.lower())\n    return cleaned == cleaned[::-1]\n```",
    "gen-03": "```python\ndef sort_names_by_score(data):\n    return [d['name'] for d in sorted(data, key=lambda x: x['score'], reverse=True)]\n```",
    "gen-04": "```python\ndef merge_sorted(a, b):\n    i, j, out = 0, 0, []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n    return out\n```",
}


def answer_task(task: dict) -> str:
    return ANSWERS.get(task["task_id"], "no answer")
