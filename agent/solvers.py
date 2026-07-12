"""Deterministic 0-token answers for math and logic.

The ranking metric is total tokens through the proxy, gated on accuracy. Any
task we can answer correctly in Python never touches the model, so it costs
zero tokens AND can't be got wrong by a weak model.

Scope check (official samples): the real tasks are mostly MULTI-STEP, which
these solvers deliberately refuse -- so they're a safe bonus for the trivial
cases that do appear, not the primary answer path. Coverage was traded away
for precision on purpose.

The hard rule here is **precision over coverage**: a confidently-wrong 0-token
answer is worse than paying the model, because it drags down the accuracy gate
that decides whether we rank at all. So every solver returns a `str` only when
it has parsed the task unambiguously, and `None` the moment anything is unclear
-- ties, cycles, missing entities, unparseable numbers, multi-step phrasing.
On `None`, agent/core.py falls back to the (strong) model.

Two entry points, both `(prompt: str) -> str | None`:
  - solve_math   : percent-of, discount, even split, powers, average speed,
                   and clean arithmetic expressions.
  - solve_logic  : transitive comparison / spatial ordering puzzles. Leaves
                   syllogisms to the model.
"""

import ast
import operator
import re

# --- safe arithmetic --------------------------------------------------------
# Never eval() user text. Walk a whitelisted AST instead: numbers and the five
# arithmetic ops only. Anything else (a Name, a Call, an attribute) returns
# None. Magnitudes are capped so a "10**9**9" style prompt can't hang the run.
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_MAX_MAGNITUDE = 10**12  # results larger than this are refused, not returned
_MAX_POW_BASE = 1000     # caps on exponentiation to bound compute
_MAX_POW_EXP = 64


def _eval_node(node):
    """Recursively evaluate a whitelisted arithmetic AST node, or return None
    for anything outside the whitelist or over the magnitude caps."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):  # bool is an int subclass; reject it
            return None
        if isinstance(node.value, (int, float)) and abs(node.value) <= _MAX_MAGNITUDE:
            return node.value
        return None
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        val = _eval_node(node.operand)
        return None if val is None else _UNARYOPS[type(node.op)](val)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Pow):
            if abs(left) > _MAX_POW_BASE or abs(right) > _MAX_POW_EXP:
                return None
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            return None
        try:
            result = _BINOPS[type(node.op)](left, right)
        except (ArithmeticError, ValueError):
            return None
        if isinstance(result, (int, float)) and abs(result) <= _MAX_MAGNITUDE:
            return result
        return None
    return None


def _safe_eval(expr: str):
    """Evaluate a bare arithmetic expression string, or None if it isn't one."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    return _eval_node(tree.body)


# --- number formatting ------------------------------------------------------

def _fmt_num(n) -> str:
    """Integer-valued floats print without a trailing '.0'; others round to a
    sensible precision (arithmetic answers are usually exact or clean)."""
    if isinstance(n, float):
        if n.is_integer():
            return str(int(n))
        return str(round(n, 4))
    return str(n)


def _fmt_money(n) -> str:
    if isinstance(n, float) and n.is_integer():
        return f"${int(n)}"
    return f"${n:.2f}"


_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def _parse_count(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return _WORD_NUMBERS.get(token)


# ============================================================================
# MATH
# ============================================================================

# Signals that a problem needs MORE than one operation. The official sample
# tasks are multi-step (sell 37%, restock, sell again; scale a recipe THEN
# price it) and a partial match would return a confidently-wrong answer --
# e.g. the bare-expression matcher once grabbed the "3/4" out of a recipe
# problem and answered "0.75". Any of these signals => every math solver
# defers to the model. Missing a 0-token win is cheap; a wrong shortcut costs
# the accuracy gate.
_MULTISTEP = re.compile(
    r"\b(then|after|afterwards|next|later|followed by|q[1-4]|"
    r"restocks?|remains?|remaining|left(?: over)?|"
    r"total(?: cost| price| amount)?|in total|altogether|combined|overall)\b",
    re.I,
)


def _too_complex(prompt: str) -> bool:
    """More than one question, or any multi-step signal: not solver territory."""
    return prompt.count("?") >= 2 or bool(_MULTISTEP.search(prompt))


_RE_PERCENT_OF = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", re.I)
_RE_POWER = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:to the power of|raised to(?: the power of)?|\*\*|\^)\s*(\d+(?:\.\d+)?)",
    re.I,
)
_RE_MONEY = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
_RE_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
# A contiguous symbolic arithmetic expression with at least one operator.
_RE_EXPR = re.compile(r"\d+(?:\.\d+)?(?:\s*[+\-*/^]\s*\d+(?:\.\d+)?)+")
# Words that signal a second operation the symbolic match won't capture, e.g.
# "17 * 23 and then subtract 40". Their presence makes a bare expression unsafe.
_RE_EXTRA_OP = re.compile(
    r"\b(subtract|minus|plus|add|added|times|multiply|multiplied|divide|divided|"
    r"increase|decrease|less|more)\b",
    re.I,
)
_DISCOUNT_WORDS = re.compile(r"\b(discount(?:ed)?|sale price|% off|percent off|marked down)\b", re.I)
_SPLIT_WORDS = re.compile(r"\b(split|evenly|each pay|per person|divided (?:among|between)|share)\b", re.I)


def _solve_percent_of(prompt: str) -> str | None:
    # Only a short, direct ask ("What is 15% of 240?"). Embedded in longer
    # prose, "X% of Y" is usually one step of a bigger problem.
    if len(prompt) > 60:
        return None
    m = _RE_PERCENT_OF.search(prompt)
    if not m:
        return None
    pct, whole = float(m.group(1)), float(m.group(2))
    return _fmt_num(pct / 100.0 * whole)


def _solve_power(prompt: str) -> str | None:
    m = _RE_POWER.search(prompt)
    if not m:
        return None
    result = _safe_eval(f"{m.group(1)}**{m.group(2)}")
    return None if result is None else _fmt_num(result)


def _solve_discount(prompt: str) -> str | None:
    if not _DISCOUNT_WORDS.search(prompt):
        return None
    # A second percentage or markup language means multiple pricing steps
    # (e.g. "marks up 40%... discounted by 25%"): grabbing one percent would
    # compute a confidently-wrong price. Single-step discounts only.
    if len(_RE_PERCENT.findall(prompt)) != 1:
        return None
    if re.search(r"\bmark(?:s|ed)?[\s-]*up\b|\bmarkup\b", prompt, re.I):
        return None
    price_m = _RE_MONEY.search(prompt)
    pct_m = _RE_PERCENT.search(prompt)
    if not price_m or not pct_m:
        return None
    price, pct = float(price_m.group(1)), float(pct_m.group(1))
    if not 0 <= pct <= 100:
        return None
    return _fmt_money(price * (1 - pct / 100.0))


def _solve_split(prompt: str) -> str | None:
    if not _SPLIT_WORDS.search(prompt):
        return None
    amount_m = _RE_MONEY.search(prompt)
    if not amount_m:
        return None
    # The number of people: a digit or number-word immediately before a
    # people-noun, e.g. "Three friends", "4 people".
    count_m = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
        r"(?:friends|people|persons|colleagues|coworkers|ways|guests|diners)\b",
        prompt, re.I,
    )
    if not count_m:
        return None
    count = _parse_count(count_m.group(1))
    if not count:
        return None
    amount = float(amount_m.group(1))
    # "rounded to the nearest cent" is the common ask; always show cents so the
    # answer reads as money regardless.
    return f"${amount / count:.2f}"


def _solve_average_speed(prompt: str) -> str | None:
    if not re.search(r"\bspeed\b", prompt, re.I):
        return None
    # Mixed-unit durations ("2 hours 15 minutes"): the single time regex would
    # grab only the first unit and compute a wrong speed. Defer.
    if re.search(r"\bhours?\b", prompt, re.I) and re.search(r"\bmin(?:ute)?s?\b", prompt, re.I):
        return None
    dist_m = re.search(
        r"(\d+(?:\.\d+)?)\s*(kilometres?|kilometers?|km|miles?|mi|metres?|meters?|m)\b",
        prompt, re.I,
    )
    time_m = re.search(
        r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
        prompt, re.I,
    )
    if not dist_m or not time_m:
        return None
    dist = float(dist_m.group(1))
    time_val = float(time_m.group(1))
    if time_val == 0:
        return None
    time_unit = time_m.group(2).lower().rstrip("s")  # normalise plural
    if time_unit in ("hour", "hr"):
        hours = time_val
    elif time_unit in ("minute", "min"):
        hours = time_val / 60.0
    elif time_unit in ("second", "sec"):
        hours = time_val / 3600.0
    else:
        return None
    dist_unit = dist_m.group(2).lower().rstrip("s")
    unit_label = "mph" if dist_unit in ("mile", "mi") else "km/h"
    return f"{_fmt_num(dist / hours)} {unit_label}"


def _solve_expression(prompt: str) -> str | None:
    # Only safe when the prompt essentially IS the expression ("Compute
    # 144 / 12."): short, exactly one symbolic expression, and no words that
    # imply a further operation. A fraction inside a word problem ("3/4 cup of
    # sugar for 12 cookies...") must never fire.
    if len(prompt) > 80 or _RE_EXTRA_OP.search(prompt):
        return None
    matches = _RE_EXPR.findall(prompt)
    if len(matches) != 1:
        return None
    result = _safe_eval(matches[0].replace("^", "**"))
    return None if result is None else _fmt_num(result)


_MATH_SOLVERS = (
    _solve_percent_of,
    _solve_discount,
    _solve_split,
    _solve_average_speed,
    _solve_power,
    _solve_expression,
)


def solve_math(prompt: str) -> str | None:
    """Answer an arithmetic word problem deterministically, or None to defer to
    the model. First confident solver wins; order matters (percent-of and
    discount are checked before the generic expression matcher).

    Ultra-conservative by design: real tasks are usually multi-step, and every
    solver defers the moment the prompt shows a second operation or question.
    Solvers are a safe 0-token bonus for the trivial cases, not the main path.
    """
    if not prompt or _too_complex(prompt):
        return None
    for solver in _MATH_SOLVERS:
        try:
            answer = solver(prompt)
        except Exception:
            answer = None  # a solver bug must never crash the task
        if answer is not None:
            return answer
    return None


# ============================================================================
# LOGIC (transitive ordering / spatial)
# ============================================================================

# Comparatives that mean left > right (the left operand ranks higher).
_GREATER = (
    r"taller|older|faster|larger|bigger|heavier|greater|stronger|longer|higher|"
    r"richer|wealthier|warmer|hotter|more"
)
# Comparatives that mean left < right.
_LESSER = (
    r"shorter|younger|slower|smaller|lighter|weaker|lower|colder|cooler|less|fewer|poorer"
)

_RE_GREATER = re.compile(rf"([A-Z][a-zA-Z]*)\s+(?:is\s+|was\s+)?(?:{_GREATER})\s+than\s+([A-Z][a-zA-Z]*)")
_RE_LESSER = re.compile(rf"([A-Z][a-zA-Z]*)\s+(?:is\s+|was\s+)?(?:{_LESSER})\s+than\s+([A-Z][a-zA-Z]*)")
_RE_BEAT = re.compile(r"([A-Z][a-zA-Z]*)\s+(?:beat|beats|beaten|defeated|outran|outscored|won against)\s+([A-Z][a-zA-Z]*)")
# "X is north of Y" (compass directions take "of").
_RE_SPATIAL = re.compile(r"([A-Z][a-zA-Z]*)\s+(?:is\s+|was\s+)?(north|south|east|west)\s+of\s+([A-Z][a-zA-Z]*)")
# "X is above Y" (above/below don't take "of").
_RE_ABOVE = re.compile(r"([A-Z][a-zA-Z]*)\s+(?:is\s+|was\s+)?(above|below|over|under)\s+([A-Z][a-zA-Z]*)")
# north/east/above treated as the "greater" direction; south/west/below as "lesser".
_SPATIAL_GREATER = {"north", "east", "above", "over"}

# Question targets.
_MAX_WORDS = re.compile(
    r"\b(tallest|oldest|largest|biggest|fastest|highest|heaviest|strongest|longest|"
    r"first|winner|won|northernmost|easternmost|furthest north|furthest east)\b", re.I,
)
_MIN_WORDS = re.compile(
    r"\b(shortest|youngest|smallest|slowest|lowest|lightest|weakest|"
    r"last|came last|loser|southernmost|westernmost|furthest south|furthest west)\b", re.I,
)
_ORDINALS = {
    "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
}
_RE_ORDINAL = re.compile(r"\b(second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th)\b", re.I)
# Syllogisms and yes/no deductions: not safe to solve by ordering. Defer.
_SYLLOGISM = re.compile(r"\b(can we conclude|does (?:it|this) follow|conclude that|all\s+\w+\s+are)\b", re.I)
# Negation, exceptions, and equality break the simple ">" graph: "Alice is NOT
# taller than Bob" would still match the comparative regex and add a WRONG
# edge. Any of these words => defer to the model.
_LOGIC_DEFER = re.compile(
    r"\b(not|except|unless|neither|nor|same|equal(?:ly)?|tie[ds]?|as \w+ as)\b|n't",
    re.I,
)


def _collect_edges(prompt: str) -> list[tuple[str, str]] | None:
    """Return directed edges (a, b) meaning a > b, or None if a relation is
    internally contradictory in direction parsing."""
    edges: list[tuple[str, str]] = []
    for a, b in _RE_GREATER.findall(prompt):
        edges.append((a, b))
    for a, b in _RE_LESSER.findall(prompt):
        edges.append((b, a))  # a < b  ->  b > a
    for a, b in _RE_BEAT.findall(prompt):
        edges.append((a, b))
    for a, direction, b in _RE_SPATIAL.findall(prompt):
        if direction.lower() in _SPATIAL_GREATER:
            edges.append((a, b))
        else:
            edges.append((b, a))
    for a, direction, b in _RE_ABOVE.findall(prompt):
        if direction.lower() in _SPATIAL_GREATER:
            edges.append((a, b))
        else:
            edges.append((b, a))
    return edges


def _total_order(edges: list[tuple[str, str]]) -> list[str] | None:
    """Topologically sort into a single strict chain (max first). Returns None
    if the order isn't unique: a tie (two sources at once), a cycle, or a
    disconnected/underspecified set all fail closed."""
    nodes = {n for e in edges for n in e}
    if len(nodes) < 2:
        return None
    successors: dict[str, set[str]] = {n: set() for n in nodes}
    indegree: dict[str, int] = {n: 0 for n in nodes}
    seen_pairs = set()
    for a, b in edges:
        if a == b:
            return None  # self-comparison: contradictory
        if (b, a) in seen_pairs:
            return None  # both a>b and b>a: contradiction
        if (a, b) in seen_pairs:
            continue  # duplicate statement, harmless
        seen_pairs.add((a, b))
        if b not in successors[a]:
            successors[a].add(b)
            indegree[b] += 1

    order: list[str] = []
    frontier = [n for n in nodes if indegree[n] == 0]
    while frontier:
        if len(frontier) > 1:
            return None  # ambiguous: more than one candidate for this rank
        current = frontier.pop()
        order.append(current)
        for nxt in successors[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                frontier.append(nxt)
    if len(order) != len(nodes):
        return None  # a cycle left nodes unplaced
    return order


def solve_logic(prompt: str) -> str | None:
    """Answer a transitive ordering / spatial puzzle, or None to defer.

    Precision-first: any ambiguity (non-unique order, missing target word,
    ordinal without a direction, out-of-range index, syllogism, negation,
    exception, or equality wording) returns None.
    """
    if not prompt or _SYLLOGISM.search(prompt) or _LOGIC_DEFER.search(prompt):
        return None

    edges = _collect_edges(prompt)
    if not edges:
        return None
    order = _total_order(edges)  # max (tallest/oldest/first) first
    if not order:
        return None

    wants_max = bool(_MAX_WORDS.search(prompt))
    wants_min = bool(_MIN_WORDS.search(prompt))
    if wants_max == wants_min:
        return None  # neither, or contradictory both

    ord_m = _RE_ORDINAL.search(prompt)
    if ord_m:
        n = _ORDINALS[ord_m.group(1).lower()]
        if n > len(order):
            return None
        # nth from the top (max side) or bottom (min side).
        return order[n - 1] if wants_max else order[-n]

    return order[0] if wants_max else order[-1]
